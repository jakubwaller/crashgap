"""Datenschutzerklärung - static legal page, reachable from the dashboard footer."""

import streamlit as st

st.set_page_config(page_title="Datenschutz - CrashGap", page_icon="🚗", layout="centered")

st.title("Datenschutzerklärung")
st.markdown("""
## Verantwortlicher

Jakub Waller, c/o IP-Management #11204, Ludwig-Erhard-Str. 18, 20459 Hamburg, E-Mail:
[crashgap@jakubwaller.eu](mailto:crashgap@jakubwaller.eu)

## Grundsatz

Diese Website ist ein statistisches Dashboard: keine Konten, keine Formulare, keine Analyse-
oder Tracking-Cookies, keine serverseitige Datenbank mit Besucherdaten. Die dargestellten
Zahlen stammen aus öffentlichen US-Unfalldatenbanken und enthalten keine Daten von Besuchern
dieser Seite.

## Hosting / Server-Logs

Die Website wird auf einem privat betriebenen Server in einem EU-Rechenzentrum gehostet. Beim
Aufruf werden technisch notwendige Daten verarbeitet (IP-Adresse, Zeitpunkt, abgerufene
Datei). Rechtsgrundlage: Art. 6 Abs. 1 lit. f DSGVO.

## Cloudflare (Auslieferung & Schutz)

Die Auslieferung erfolgt über Cloudflare, Inc. (Content Delivery & DDoS-Schutz). Cloudflare
verarbeitet dabei die IP-Adressen der Besucher, um die Seite auszuliefern und zu schützen, und
setzt ggf. ein technisch notwendiges Anti-Bot-Cookie — kein Profiling, keine Einwilligung
erforderlich. Cloudflare ist nach dem EU-US Data Privacy Framework zertifiziert; ein
Auftragsverarbeitungsvertrag (AVV) besteht.

## Kontakt per E-Mail

Bei Kontakt per E-Mail werden die übermittelten Angaben nur zur Bearbeitung des Anliegens
verarbeitet (Art. 6 Abs. 1 lit. f DSGVO) und nicht weitergegeben.

## Deine Rechte

Auskunft (Art. 15), Berichtigung (Art. 16), Löschung (Art. 17), Einschränkung (Art. 18),
Datenübertragbarkeit (Art. 20), Widerspruch (Art. 21 DSGVO). Kontakt:
[crashgap@jakubwaller.eu](mailto:crashgap@jakubwaller.eu).

[← Zum Dashboard](/)
""")
