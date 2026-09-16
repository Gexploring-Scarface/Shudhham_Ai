import os

import streamlit as st

from rag import WasteRAG, ONTOLOGY
from granite_client import GraniteClient


st.set_page_config(
    page_title="Shuddham.AI",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 10% 0%, #0e211b 0, #07100d 36%, #050907 100%);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 980px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-top: 5px;
    }

    .logo {
        width: 56px;
        height: 56px;
        border-radius: 17px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #11a56f, #08794f);
        font-size: 28px;
        box-shadow: 0 12px 32px rgba(17, 165, 111, .24);
    }

    .title {
        font-size: 2.45rem;
        font-weight: 850;
        letter-spacing: -1.2px;
        color: #f2fff9;
    }

    .tag {
        color: #9eb6ac;
        font-size: 1rem;
        margin: 7px 0 24px 71px;
    }

    .hero {
        padding: 22px 24px;
        border: 1px solid #214a3a;
        background: rgba(12, 28, 23, .82);
        border-radius: 22px;
        margin: 18px 0;
    }

    .hero-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #effff8;
        margin-bottom: 5px;
    }

    .muted {
        color: #91aaa0;
    }

    .mini {
        font-size: .84rem;
        color: #8fa59c;
    }

    .chip {
        display: inline-block;
        background: #daf7ea;
        color: #08684e;
        padding: 5px 10px;
        border-radius: 999px;
        font-weight: 800;
        font-size: .80rem;
        margin: 4px 5px 0 0;
    }

    .result {
        border: 1px solid #285845;
        background: linear-gradient(180deg, #0d1d17, #0a1713);
        border-radius: 22px;
        padding: 23px;
        margin-top: 18px;
    }

    .result-title {
        font-size: 1.65rem;
        font-weight: 850;
        color: #f3fff9;
        margin-top: 10px;
    }

    .big-answer {
        font-size: 1.05rem;
        color: #dcece5;
        line-height: 1.65;
        margin-top: 8px;
    }

    .confidence {
        font-size: .9rem;
        margin-top: 14px;
        color: #9cb7ac;
    }

    .evidence {
        padding: 12px 14px;
        border: 1px solid #1f4034;
        border-radius: 14px;
        margin: 9px 0;
        background: #08130f;
    }

    .section {
        border: 1px solid #1b382f;
        border-radius: 20px;
        padding: 18px 20px;
        margin-top: 15px;
        background: rgba(7, 18, 14, .75);
    }

    .section h3 {
        color: #effff8;
        font-size: 1.05rem;
        margin: 0 0 9px;
    }

    .ok {
        color: #79e0b7;
        font-weight: 800;
    }

    .warn {
        color: #ffd37e;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Initialize application components
# -------------------------------------------------------------------

rag = WasteRAG()
granite = GraniteClient()


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.markdown(
    '<div class="brand">'
    '<div class="logo">♻️</div>'
    '<div class="title">Shuddham.AI</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="tag">'
    'AI-assisted waste segregation for cleaner, more sustainable communities.'
    '</div>',
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Hero section
# -------------------------------------------------------------------

st.markdown(
    '<div class="hero">'
    '<div class="hero-title">What are you disposing?</div>'
    '<div class="muted">Describe any object or waste item in natural language.</div>'
    '</div>',
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# User input
# -------------------------------------------------------------------

query = st.text_input(
    "Waste item",
    placeholder=(
        "Try: old laptop, tea leaves, broken keyboard, "
        "plastic packet, medicine"
    ),
    label_visibility="collapsed",
)


examples = [
    "old laptop",
    "banana peel",
    "plastic bag",
    "broken keyboard",
    "used cooking oil",
    "old clothes",
]

cols = st.columns(len(examples))

for i, example in enumerate(examples):
    with cols[i]:
        st.caption(example)


# -------------------------------------------------------------------
# Analyze controls
# -------------------------------------------------------------------

left, right = st.columns([1, 1])

with left:
    analyze = st.button(
        "Analyze with AI",
        use_container_width=True,
        type="primary",
    )

with right:
    status = (
        "IBM Granite connected"
        if granite.enabled
        else "RAG + Granite-ready demo mode"
    )
    st.caption(status)


# -------------------------------------------------------------------
# Main analysis pipeline
# -------------------------------------------------------------------

if analyze:

    # ---------------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------------

    if not query or not query.strip():
        st.warning("Enter a waste item first.")
        st.stop()

    query_clean = query.strip()

    # ---------------------------------------------------------------
    # Step 1: Generic ontology classification
    # ---------------------------------------------------------------

    try:
        generic = rag.classify_generic(query_clean)
    except Exception as exc:
        generic = None
        st.caption(f"Local classification skipped: {exc}")

    # IMPORTANT:
    # generic can legitimately be None if no confident ontology
    # classification was found.
    #
    # We therefore extract the key safely instead of doing:
    # generic["key"]

    if isinstance(generic, dict):
        generic_key = generic.get("key")
    else:
        generic_key = None
        generic = None

    # ---------------------------------------------------------------
    # Step 2: Granite classification when local ontology is uncertain
    # ---------------------------------------------------------------

    granite_class = None

    if generic is None and granite.enabled:

        category_options = {
            k: v.get("label", k)
            for k, v in ONTOLOGY.items()
            if isinstance(v, dict)
        }

        try:
            granite_class = granite.classify(
                query_clean,
                category_options,
            )

            if (
                isinstance(granite_class, dict)
                and granite_class.get("category_key") in ONTOLOGY
            ):
                try:
                    granite_confidence = int(
                        granite_class.get("confidence", 0)
                    )
                except (TypeError, ValueError):
                    granite_confidence = 0

                if granite_confidence >= 55:
                    generic = {
                        "key": granite_class["category_key"],
                        "matched_alias": None,
                        "score": granite_confidence / 100,
                        "method": "IBM Granite semantic classification",
                    }

                    generic_key = generic["key"]

        except Exception as exc:
            st.info(
                "Granite classification unavailable; continuing with "
                f"the local reasoning layer. ({exc})"
            )

    # ---------------------------------------------------------------
    # Step 3: RAG search
    # ---------------------------------------------------------------

    preferred = generic_key

    try:
        results = rag.search(
            query_clean,
            k=4,
            preferred_category=preferred,
        )
    except Exception as exc:
        results = []
        st.error(f"Knowledge-base search failed: {exc}")

    # ---------------------------------------------------------------
    # Step 4: Handle case where nothing can be identified
    # ---------------------------------------------------------------

    if not generic and not results:

        st.error(
            "I couldn't identify this item reliably. "
            "Try adding material or context, such as "
            "'plastic toy', 'electronic device', or 'food waste'."
        )

    else:

        # -----------------------------------------------------------
        # Find best RAG result
        # -----------------------------------------------------------

        best = results[0] if results else None

        # If ontology classification exists, prefer a KB item from
        # that category.
        if generic and results:

            same = [
                d
                for d in results
                if d.get("category_key") == generic_key
            ]

            if same:
                best = same[0]

        # -----------------------------------------------------------
        # Display result
        # -----------------------------------------------------------

        if best:

            # -------------------------------------------------------
            # Confidence
            # -------------------------------------------------------

            try:
                confidence = rag.confidence(best, generic)
            except Exception:
                # Safe fallback if confidence() does not handle None
                # or malformed classification data.
                confidence = 0

            try:
                confidence = max(
                    0,
                    min(100, int(round(float(confidence))))
                )
            except (TypeError, ValueError):
                confidence = 0

            # -------------------------------------------------------
            # Determine category label safely
            # -------------------------------------------------------

            if generic and generic.get("key"):

                ontology_entry = ONTOLOGY.get(
                    generic["key"],
                    {},
                )

                if isinstance(ontology_entry, dict):
                    label = ontology_entry.get(
                        "label",
                        best.get(
                            "category",
                            "Other / Unclassified Waste",
                        ),
                    )
                else:
                    label = best.get(
                        "category",
                        "Other / Unclassified Waste",
                    )

            else:

                label = best.get(
                    "category",
                    "Other / Unclassified Waste",
                )

            # -------------------------------------------------------
            # Determine match explanation
            # -------------------------------------------------------

            if generic:
                matched = (
                    generic.get("matched_alias")
                    or generic.get("method")
                    or "semantic understanding"
                )
            else:
                matched = "retrieval-based understanding"

            # -------------------------------------------------------
            # Result card
            # -------------------------------------------------------

            st.markdown(
                '<div class="result">',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<span class="chip">{label}</span>',
                unsafe_allow_html=True,
            )

            item_name = best.get(
                "item",
                query_clean,
            )

            st.markdown(
                f'<div class="result-title">'
                f'{str(item_name).title()}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="big-answer">'
                f'<b>Recommended disposal:</b> '
                f'{best.get("bin", "Follow local waste collection rules.")}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="big-answer">'
                f'<b>What to do:</b> '
                f'{best.get("action", "Follow local authority guidance.")}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="big-answer">'
                f'<b>Safety:</b> '
                f'{best.get("safety", "Handle and dispose of safely.")}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # -------------------------------------------------------
            # Confidence level
            # -------------------------------------------------------

            if confidence >= 80:
                level = "High"
                cls = "ok"
            elif confidence >= 50:
                level = "Moderate"
                cls = "warn"
            else:
                level = "Low"
                cls = "warn"

            st.markdown(
                f'<div class="confidence">'
                f'Identification confidence: '
                f'<span class="{cls}">'
                f'{confidence}% · {level}'
                f'</span> · {matched}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # -------------------------------------------------------
            # Granite grounded response
            # -------------------------------------------------------

            if granite.enabled:

                try:

                    context_parts = []

                    for d in results:

                        context_parts.append(
                            f"ITEM={d.get('item', '')} | "
                            f"CATEGORY={d.get('category', '')} | "
                            f"DISPOSAL={d.get('bin', '')} | "
                            f"ACTION={d.get('action', '')} | "
                            f"SAFETY={d.get('safety', '')}"
                        )

                    context = "\n".join(context_parts)

                    system = (
                        "You are Shuddham.AI. "
                        "Rewrite the grounded recommendation using ONLY "
                        "the supplied retrieved evidence. "
                        "Never invent disposal rules. "
                        "Mention that local authority rules may differ."
                    )

                    prompt = (
                        f"User item: {query_clean}\n"
                        f"Retrieved evidence:\n{context}\n"
                        "Return 3 concise sentences."
                    )

                    live = granite.chat(
                        system,
                        prompt,
                        max_tokens=220,
                    )

                    if live:

                        st.divider()

                        st.markdown(
                            "**IBM Granite · grounded response**"
                        )

                        st.write(live)

                except Exception as exc:

                    st.caption(
                        f"Granite generation skipped: {exc}"
                    )

            # -------------------------------------------------------
            # RAG evidence
            # -------------------------------------------------------

            with st.expander("See the RAG evidence"):

                st.caption(
                    "The retrieval layer supplies the factual disposal "
                    "guidance used by the AI response."
                )

                for d in results:

                    cosine_score = d.get("cosine_score", 0)
                    overlap = d.get("overlap", 0)

                    try:
                        cosine_score = float(cosine_score)
                    except (TypeError, ValueError):
                        cosine_score = 0.0

                    item = str(
                        d.get("item", "Unknown item")
                    ).title()

                    action = d.get(
                        "action",
                        "No action guidance available.",
                    )

                    st.markdown(
                        f'<div class="evidence">'
                        f'<b>{item}</b><br>'
                        f'<span class="mini">'
                        f'Similarity {cosine_score:.3f} · '
                        f'token overlap {overlap}'
                        f'</span><br>'
                        f'<span class="muted">{action}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown(
                '</div>',
                unsafe_allow_html=True,
            )

        # -----------------------------------------------------------
        # Generic classification exists but there is no KB result
        # -----------------------------------------------------------

        else:

            if generic and generic.get("key"):

                ontology_entry = ONTOLOGY.get(
                    generic["key"],
                    {},
                )

                if isinstance(ontology_entry, dict):
                    category_label = ontology_entry.get(
                        "label",
                        "Other / Unclassified Waste",
                    )
                else:
                    category_label = "Other / Unclassified Waste"

            else:

                category_label = "Other / Unclassified Waste"

            st.warning(
                f"The system identified this as "
                f"**{category_label}**, but the knowledge base does not "
                f"yet contain enough disposal guidance for this specific "
                f"item. Follow your local collection rules."
            )


# -------------------------------------------------------------------
# How Shuddham.AI works
# -------------------------------------------------------------------

st.markdown(
    '<div class="section">'
    '<h3>How Shuddham.AI works</h3>'
    '<div class="muted">'
    '1. Understand the item → '
    '2. classify it against a broad waste ontology → '
    '3. retrieve relevant disposal evidence with RAG → '
    '4. use IBM Granite to produce a concise grounded explanation.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Responsible AI
# -------------------------------------------------------------------

st.markdown(
    '<div class="section">'
    '<h3>Responsible AI</h3>'
    '<div class="muted">'
    'The app avoids forced classifications when evidence is weak, '
    'exposes retrieved evidence, does not require personal data, and '
    'treats recommendations as decision support. Local waste-management '
    'authority guidance takes precedence.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------

st.markdown(
    '<div class="mini" style="margin-top:16px">'
    'Primary SDG 12 · Secondary SDG 11 · Shuddham.AI prototype'
    '</div>',
    unsafe_allow_html=True,
)