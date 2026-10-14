import re

# Build-spec values are interpolated into a text template handed to
# execute_text_template(): this.DeRef('{in_model}') and the SELECT column list.
# That DSL uses {% %} / {{ }} directives and single-quoted literals, so a value
# with a quote, brace, percent, backslash or whitespace can break out and inject
# template/SQL (RUD-3049, same class as RUD-3028).

# Model/input references are slash-separated identifier segments, e.g.
# 'inputs/tbl_a', 'models/input_table_size', 'entity/user/var_table'.
_MODEL_REF_RE = re.compile(r"[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)*")


def validate_model_ref(ref: str) -> str:
    if not isinstance(ref, str) or _MODEL_REF_RE.fullmatch(ref) is None:
        raise ValueError(
            f"Invalid input/model reference {ref!r}: expected slash-separated "
            "identifier segments (e.g. 'inputs/tbl_a', 'entity/user/var')"
        )
    return ref


def validate_sql_identifier(name: str) -> str:
    # Column names from get_columns() are interpolated raw into the SELECT list.
    # str.isidentifier() accepts a bare column identifier (incl. non-ASCII, matching
    # attribution_model._validate_column_name) and rejects operators, quotes,
    # braces, percent signs and whitespace.
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(
            f"Invalid column identifier {name!r}: expected a bare column identifier"
        )
    return name
