import pytest

from copilot.ingest.models import Unit
from copilot.retrieve.chunks import Corpus


@pytest.fixture
def make_unit():
    def build(
        id,
        text,
        *,
        regulation="aiact",
        language="en",
        type="paragraph",
        title=None,
        article=None,
        annex=None,
        parent_id=None,
    ) -> Unit:
        return Unit(
            id=id,
            regulation=regulation,
            language=language,
            type=type,
            article=article,
            annex=annex,
            number=id.rsplit(":", 1)[-1],
            title=title,
            chapter=None,
            text=text,
            refs=(),
            parent_id=parent_id,
        )

    return build


@pytest.fixture
def corpus(make_unit):
    return Corpus(
        [
            make_unit(
                "aiact:art:6:2",
                "AI systems referred to in Annex III shall be considered high-risk.",
                title="Classification rules for high-risk AI systems",
                article="6",
            ),
            make_unit(
                "aiact:art:5:1",
                "The following AI practices shall be prohibited: social scoring.",
                title="Prohibited AI practices",
                article="5",
            ),
            make_unit(
                "gdpr:art:22:1",
                "The data subject shall have the right not to be subject to a decision based solely"
                " on automated processing.",
                title="Automated individual decision-making",
                regulation="gdpr",
                article="22",
            ),
            make_unit(
                "gdpr:art:33:1",
                "The controller shall notify the data breach to the authority within 72 hours.",
                title="Notification of a personal data breach",
                regulation="gdpr",
                article="33",
            ),
            # French and German copies, so the language paths have something to retrieve
            make_unit(
                "aiact:art:6:2",
                "Les systèmes d'IA visés à l'annexe III sont considérés comme à haut risque.",
                title="Règles de classification des systèmes d'IA à haut risque",
                language="fr",
                article="6",
            ),
            make_unit(
                "aiact:art:5:1",
                "Les pratiques d'IA suivantes sont interdites : la notation sociale.",
                title="Pratiques interdites en matière d'IA",
                language="fr",
                article="5",
            ),
            make_unit(
                "gdpr:art:33:1",
                "Le responsable notifie la violation à l'autorité dans les 72 heures.",
                title="Notification d'une violation de données",
                regulation="gdpr",
                language="fr",
                article="33",
            ),
            make_unit(
                "aiact:art:6:2",
                "Die in Anhang III genannten KI-Systeme gelten als hochriskant.",
                title="Klassifizierungsvorschriften für Hochrisiko-KI-Systeme",
                language="de",
                article="6",
            ),
            make_unit(
                "aiact:art:5:1",
                "Folgende KI-Praktiken sind verboten: das soziale Scoring.",
                title="Verbotene Praktiken im KI-Bereich",
                language="de",
                article="5",
            ),
            make_unit(
                "gdpr:art:33:1",
                "Der Verantwortliche meldet die Verletzung der Aufsichtsbehörde binnen 72 Stunden.",
                title="Meldung von Verletzungen",
                regulation="gdpr",
                language="de",
                article="33",
            ),
        ]
    )
