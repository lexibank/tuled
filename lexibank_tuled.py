from pathlib import Path
import attr
from csvw import Datatype
from pylexibank import Concept, Language, Cognate, Lexeme
from pylexibank.dataset import Dataset as BaseDataset
from pylexibank.util import progressbar
from pyclts import CLTS
from lingpy import Multiple, Wordlist
from lingpy.basictypes import lists


@attr.s
class CustomConcept(Concept):
    Number = attr.ib(default=None)
    Portuguese_Gloss = attr.ib(default=None)
    EOL_ID = attr.ib(default=None)
    Semantic_Field = attr.ib(default=None)


@attr.s
class CustomLanguage(Language):
    Latitude = attr.ib(default=None)
    Longitude = attr.ib(default=None)
    SubGroup = attr.ib(default=None)
    Source = attr.ib(default=None)


@attr.s
class CustomCognate(Cognate):
    Segment_Slice = attr.ib(default=None)


@attr.s
class Form(Lexeme):
    Morphemes = attr.ib(default=None)
    SimpleCognate = attr.ib(default=None)
    PartialCognates = attr.ib(default=None)


class Dataset(BaseDataset):
    dir = Path(__file__).parent
    id = "tuled"
    concept_class = CustomConcept
    language_class = CustomLanguage
    cognate_class = CustomCognate
    lexeme_class = Form
    writer_options = dict(keep_languages=False, keep_parameters=False)

    def cmd_makecldf(self, args):
        from pybtex import database, errors
        errors.strict = False
        bibdata = database.parse_file(str(self.raw_dir.joinpath('sources.bib')))
        args.writer.add_sources(bibdata)
        args.writer["FormTable", "Segments"].datatype = Datatype.fromvalue(
            {"base": "string", "format": "([\\S]+)( [\\S]+)*"}
            )
        args.writer["FormTable", "Morphemes"].separator = " "
        args.writer["FormTable", "PartialCognates"].separator = " "
        concepts = args.writer.add_concepts(lookup_factory=lambda c: c.english)
        errors, blacklist = set(), set()

        languages = {}
        sources = {}
        for row in self.languages:
            if not -90 < float(row['Latitude']) < 90:
                errors.add(f'LATITUDE {row["Name"]}')
            elif not -180 < float(row['Longitude']) < 180:
                errors.add(f'LONGITUDE {row["Name"]}')
            else:
                try:
                    args.writer.add_language(
                        ID=row['ID'],
                        Name=row['Name'],
                        SubGroup=row['SubGroup'],
                        Latitude=row['Latitude'],
                        Longitude=row['Longitude'],
                        Glottocode=row['Glottocode'] if row['Glottocode'] != '???' else None,
                    )
                    languages[row['Name']] = row['ID']
                    sources[row['Name']] = []
                    for source in row['Sources'].split(','):
                        if source in bibdata.entries:
                            sources[row['Name']] += [source]
                        else:
                            errors.add(f'BIBTEX MISSING {source}')

                except ValueError:
                    errors.add(f'LANGUAGE ID {row["ID"]}')
                    args.log.warning(f'Invalid Language ID {row["ID"]}')

        wl = Wordlist(self.raw_dir.joinpath('tuled.tsv').as_posix())
        etd = wl.get_etymdict(ref='cogids')
        alignments, problems = {}, set()
        for cogid, vals in progressbar(etd.items(), desc='aligning data'):
            idxs = []
            for idx in vals:
                if idx:
                    idxs += idx
            positions = [wl[idx, 'cogids'].index(cogid) for idx in idxs]
            alms, new_idxs = [], []
            for idx, pos in zip(idxs, positions):
                try:
                    tks = lists(wl[idx, 'tokens']).n[pos]
                    if not ' '.join(tks).strip():
                        raise IndexError
                    alms += [tks]
                    new_idxs += [(idx, pos)]
                except IndexError:
                    problems.add((idx, pos))
            if alms:
                msa = Multiple(alms)
                msa.prog_align()
                for i, alm in enumerate(msa.alm_matrix):
                    alignments[new_idxs[i][0], new_idxs[i][1], cogid] = ' '.join(alm)
            else:
                errors.add(f'ALIGNMENT empty {cogid}')

        bipa = CLTS(args.clts.dir).bipa
        for idx, tokens, glosses, cogids, alignment in wl.iter_rows(
                'tokens', 'morphemes', 'cogids', 'alignment'):
            tl, gl, cl, al = (
                    len(lists(tokens).n),
                    len(glosses),
                    len(cogids),
                    len(lists(alignment).n)
                    )
            if tl != gl or tl != cl or gl != cl or al != gl or al != cl:
                errors.add(f'LENGTH: {idx} {wl[idx, "language"]} {wl[idx, "concept"]}')
                blacklist.add(idx)
            for token in tokens:
                if bipa[token].type == 'unknownsound':
                    errors.add(f'SOUND: {token}')
                    blacklist.add(idx)

        visited = set()
        for idx in wl:
            if wl[idx, 'concept'] not in concepts:
                if wl[idx, 'concept'] not in visited:
                    args.log.warning(f'Missing concept {wl[idx, "concept"]}')
                    visited.add(wl[idx, 'concept'])
                    errors.add(f'CONCEPT {wl[idx, "concept"]}')
            elif wl[idx, 'doculect'] not in languages:
                if wl[idx, 'doculect'] not in visited:
                    args.log.warning(f'Missing language {wl[idx, "doculect"]}')
                    visited.add(wl[idx, 'doculect'])
                    errors.add(f'LANGUAGE {wl[idx, "doculect"]}')
            else:
                if ''.join(wl[idx, 'tokens']).strip() and idx not in blacklist:
                    lex = args.writer.add_form_with_segments(
                        Language_ID=languages[wl[idx, 'doculect']],
                        Parameter_ID=concepts[wl[idx, 'concept']],
                        Value=wl[idx, 'value'] or ''.join(wl[idx, 'tokens']),
                        Form=wl[idx, 'form'] or ''.join(wl[idx, 'tokens']),
                        Segments=wl[idx, 'tokens'],
                        Morphemes=wl[idx, 'morphemes'],
                        SimpleCognate=wl[idx, 'cogid'],
                        PartialCognates=wl[idx, 'cogids'],
                        Source=sources[wl[idx, 'doculect']],
                    )
                    for gloss_index, cogid in enumerate(wl[idx, 'cogids']):
                        args.writer.add_cognate(
                                lexeme=lex,
                                Cognateset_ID=cogid,
                                Segment_Slice=gloss_index+1,
                                Alignment=alignments.get(
                                    (idx, gloss_index, cogid),
                                    ''),
                                Alignment_Method='SCA'
                                )
                else:
                    args.log.warning(f"Entry ID={idx}, concept={wl[idx, 'concept']}, language={wl[idx, 'doculect']} is empty")

        with open(self.dir.joinpath('errors.md'), 'w', encoding="utf-8") as f:
            f.write('# Error Analysis for TULED\n')
            for error in sorted(errors):
                f.write('* '+error+'\n')
