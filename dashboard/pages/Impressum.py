"""Impressum - static legal page, reachable from the dashboard footer."""

import streamlit as st

st.set_page_config(page_title="Impressum - CrashGap", page_icon="🚗", layout="centered")

st.title("Impressum")
st.markdown("""
Angaben gemäß § 5 DDG:

Jakub Waller\\
c/o IP-Management #11204\\
Ludwig-Erhard-Str. 18\\
20459 Hamburg

## Kontakt

E-Mail: [crashgap@jakubwaller.eu](mailto:crashgap@jakubwaller.eu)

## Verantwortlich für den Inhalt

Jakub Waller (Anschrift wie oben)

Diese Website ist ein nicht-kommerzielles Open-Source-Projekt. Datengrundlage sind die
öffentlichen Unfalldatenbanken der US-Verkehrsbehörde NHTSA (FARS, CISS, NASS-CDS; US public
domain). Die Website ist kein Angebot der NHTSA.

[← Zum Dashboard](/)
""")
