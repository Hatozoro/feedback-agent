# App Feedback Agent

Automatischer Bot zur Analyse von App Reviews für Nordkurier & Schwäbische.

## Setup für neue Maintainer
Falls der Bot nicht mehr läuft (z.B. API Key ungültig):
1. Neuen Google AI Studio Key erstellen.
2. Auf GitHub -> Settings -> Secrets -> Actions -> "GEMINI_API_KEY" aktualisieren.

## Tech Stack
- Python (Google Gemini 2.0 Flash)
- GitHub Actions (Täglicher Run um 07:00 Uhr)