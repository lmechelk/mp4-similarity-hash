masther thesis project


# TABLESTRUCTURE

PATH
Dateipfad
zB C:\Testdaten\EVA\D01_flat_yt.mp4

DB_NAME
Name der Datenbank
zB Vision, Eva, Selfmade

ID
Eindeutige Kennung des Geräts ("Device" bei Eva/Vision) oder der AI
zB D01, AI01

BRAND
Hersteller des Smartphones oder der KI
zB Samsung, Grok

MODEL
Genaue Version des Smartphones oder der KI
zB GalaxyS3Mini, Imagine_0.9

MEDIA_TYPE
Dateiendung bzw. Containerformat
zB MP4, MOV

CONTENT_SOURCE
Originäre Quelle der Videodatei um zwischen Kameras, vollständig ki-generierten Videos (mittels Text-to-Video) 
oder Videos, die auf Bild-Vorlagen basieren (Image-to-Video), zu unterscheiden
zB DIGITAL_CAMERA, AI_GENERATION, MIXED

CONTENT_TYPE
Kontext der Aufnahme
zB flat, indoor, outdoor, deepfake, synthetic

PROCESSING
Primäre Verarbeitung des Videos
zB native, youtube, whattsapp, ai

TAMPERING
Nachträgliche Veränderung des Videos (nur Eva-DB)
zB None, Avidemux

STRUCTURE_PRETTY
Gut lesbarer String der MP4-Struktur (nur nice to have)
zB moov(mvhd,trak(tkhd))

STRUCTURE_JSON
Exakte Baumstruktur als JSON-String der zur N-Gram-Generierung genutzt wird
zB [{"type":"moov","children":[{"type":"mvhd"}]}]

NGRAM_2
Kompakter Similarity Digest (String), der die Signatur des Videos auf Basis von 2-Grammen (alphabetisch, Duplikats-bereinigt) repräsentiert und jeweils 8 lückenlose Zeichen entsprechen einem 2-Gramm-Feature
zB moovmvhdtrakmvhdtraktkhd