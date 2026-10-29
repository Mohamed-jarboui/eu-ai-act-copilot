import pytest

from copilot.ingest.models import Unit


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
