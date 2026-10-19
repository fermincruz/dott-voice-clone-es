# Voces en español para Day of the Tentacle Remastered

Este proyecto reúne herramientas para crear, a partir de una copia del juego
aportada por el usuario, un `tenta.cle` que reproduce en español los diálogos, a partir de clonación de voz de cada uno de los personajes y síntesis a partir de los subtítulos en español.

El flujo extrae las voces inglesas y los subtítulos oficiales en español del
propio juego, crea referencias de voz por personaje para su clonación, sintetiza los diálogos con
VoxCPM 2 y reconstruye una copia del paquete. Se incorporan también mecanismos intermedios de control, para detectar posibles errores en las voces generadas y poder retocar los textos de los subtítulos.

El repositorio contiene las herramientas, no una copia del juego. El usuario
debe aportar su propio `tenta.cle`. No redistribuyas el juego, sus audios
extraídos ni otros materiales protegidos sin autorización.

## Qué genera

- `audio/`: MP3 y textos ingleses extraídos del juego.
- `translations_es.csv`: diálogos ingleses y sus subtítulos españoles extraídos
  de la localización del propio juego; se puede revisar antes de sintetizar.
- `voice_references/`: referencias de audio agrupadas por personaje.
- `audio_es/`: WAV españoles generados, con un nombre que permite asociarlos a
  su muestra original.
- `source/tenta_es.cle`: paquete reconstruido con las voces españolas.
- `source/tenta_es.cle.report.csv`: manifiesto con las muestras reemplazadas y
  sus duraciones.

El empaquetador sustituye los diálogos alemanes que tienen un WAV equivalente.
Deja intactos el banco inglés y los efectos de sonido alemanes.

## Requisitos

- Windows, PowerShell y Python 3.12 disponible mediante el lanzador `py`.
- `ffmpeg` instalado y accesible desde `PATH` para el empaquetado. Compruébalo
  con `ffmpeg -version`.
- Conexión a Internet durante la instalación y la primera descarga de los
  modelos de VoxCPM y Whisper.
- Se recomienda una GPU NVIDIA compatible para sintetizar el lote completo. La
  opción CPU existe, pero la síntesis puede tardar considerablemente más.
- Espacio libre para los modelos, los archivos extraídos, los WAV generados y
  el paquete final, que ocupa varios gigabytes.

## Guía paso a paso

Ejecuta los comandos desde la carpeta raíz del proyecto.

### 1. Copiar el archivo fuente del juego

Localiza `tenta.cle` en la carpeta de instalación del juego y cópialo al
directorio `source` del proyecto. Por ejemplo, para una instalación de GOG:

```powershell
New-Item -ItemType Directory -Force .\source
Copy-Item "<carpeta del juego>\tenta.cle" .\source\tenta.cle
```

Mantén el original intacto: los scripts leen `source\tenta.cle` y escriben el
paquete modificado en otro archivo.

### 2. Instalar el entorno de síntesis

En una máquina NVIDIA, elige el índice CUDA compatible con sus controladores
(el instalador admite `cu128`, `cu129` y `cu130`):

```powershell
.\setup_voxcpm_gpu.ps1 -CudaIndex cu128
```

Si no vas a usar una GPU, instala la variante CPU:

```powershell
.\setup_voxcpm_cpu.ps1
```

Los instaladores crean `.venv`, instalan las dependencias del proyecto,
PyTorch y VoxCPM 2. Los pesos se descargan la primera vez y se guardan en
`model_cache`.

### 3. Extraer las voces y los textos ingleses

```powershell
.\.venv\Scripts\python.exe .\extract_tentacle.py --overwrite
```

El script crea parejas MP3/TXT en `audio/`. Los nombres conservan las claves
del juego que se usarán para emparejar las voces en los pasos siguientes. Si
encuentra claves de subtítulo sin muestra propia entre dos muestras de voz
consecutivas del mismo grupo, las añade a la muestra anterior. También añade a
la última muestra del grupo los fragmentos huérfanos que aparecen después de
ella.

### 4. Extraer los subtítulos españoles del juego

```powershell
.\.venv\Scripts\python.exe .\extract_spanish_subtitles.py
```

Este paso lee la localización española dentro del `tenta.cle` del usuario,
genera `translations_es.csv` y crea archivos `.es.txt` junto a los MP3. Revisa
el CSV y completa o corrige la columna `spanish` antes de sintetizar. Las filas
sin texto español no se sintetizan.

La extracción aplica la misma asociación de fragmentos que el paso anterior:
los subtítulos huérfanos situados entre dos muestras de voz del mismo grupo se
unen al diálogo de la muestra anterior; los que aparecen después de la última
muestra se unen a esa última. `subtitle_merges.csv` registra cada asociación,
incluyendo la clave huérfana y la muestra siguiente, si la hay. Las claves
anteriores a la primera muestra del grupo o que sí tienen su propia muestra no
se unen automáticamente.

### 5. Crear las referencias de voz por personaje

```powershell
.\.venv\Scripts\python.exe .\prepare_voice_references.py --device cuda
```

El script identifica el personaje por el código incluido en el nombre del
audio, escoge sus clips de diálogo más largos y los concatena hasta reunir al
menos cinco segundos cuando hace falta. Excluye efectos de sonido y guarda una
referencia WAV por personaje en `voice_references/`. También genera una
transcripción automática para cada referencia. Si no hay GPU disponible, usa
`--device cpu`; la primera ejecución descarga el modelo de Whisper.

### 6. Generar los WAV en español

```powershell
.\.venv\Scripts\python.exe .\synthesize_es.py `
  --translations .\translations_es.csv `
  --voice-references .\voice_references `
  --reference-source clip `
  --device cuda `
  --output .\audio_es
```

Por defecto, `--reference-source clip` usa como referencia el MP3 inglés del
diálogo que se está doblando. Para clips de menos de un segundo, usa la
referencia WAV del personaje correspondiente en `voice_references/`; por eso
debes pasar `--voice-references` aunque la mayoría de diálogos usen su propio
audio. El modo `--cloning-mode reference` (predeterminado) pasa esta referencia
sin texto de prompt, evitando continuar la fonética inglesa. Usa
`--reference-source character` para forzar la referencia compartida del
personaje en todos los diálogos.

El script muestra el progreso y la referencia elegida para cada diálogo, e
indica cuántos clips cortos usan el fallback. Crea archivos
`EN_<clave>.es.wav` en `audio_es`.

Si vuelves a ejecutar el comando después de editar `translations_es.csv`, el
script sintetiza solo las traducciones nuevas o modificadas y los WAV que
falte generar. Compara el texto con el manifiesto
`audio_es/.translation_manifest.json`; al usarlo por primera vez, toma los
archivos `audio/<clave>.es.txt` como referencia para no regenerar los WAV
existentes cuya traducción no haya cambiado. No hace falta indicar qué filas
editaste. Usa `--overwrite` para forzar la regeneración de todos los trabajos
seleccionados.

También puedes ejecutar la síntesis en otro equipo con GPU; después copia
todos los WAV `*.es.wav` resultantes a `audio_es`, conservando sus nombres.

### 7. Revisar los audios generados

Antes de empaquetar el juego, compara la duración de cada WAV español con su
MP3 inglés correspondiente:

```powershell
.\.venv\Scripts\python.exe .\review_spanish_audio.py
```

El informe `audio_es_review.csv` marca duraciones inferiores a la mitad o
superiores al doble de la original. Ajusta esos límites con
`--min-duration-ratio` y `--max-duration-ratio`. Si el informe ya existe, usa
`--overwrite` para reemplazarlo. Por defecto, el CSV solo incluye posibles
problemas: duraciones anómalas, audios ausentes o inválidos y alertas ASR.
Añade `--report-all` si también quieres incluir las filas sin alertas.

Los casos problemáticos también se copian a `audio_es_review_cases`, en una
subcarpeta por diálogo. Cada subcarpeta contiene el MP3 inglés y su TXT, además
del WAV español y su TXT (`.es.txt`). Usa `--review-dir` para cambiar el
destino; `--overwrite` permite reemplazar copias existentes. Al repetir la
revisión, se actualizan los casos que siguen marcados y se retiran las carpetas
anteriores ya resueltas si solo contienen los cuatro archivos copiados por el
script.

Opcionalmente, activa Whisper para transcribir los WAV españoles y comparar el
texto reconocido con la columna `spanish` del CSV. De forma predeterminada,
solo aplica ASR a los audios que la revisión de duración haya marcado como
cortos o largos:

```powershell
.\.venv\Scripts\python.exe .\review_spanish_audio.py --asr --asr-device cuda --report .\audio_es_review_asr.csv
```

El modo ASR usa `openai/whisper-small` y descarga sus pesos la primera vez.
Marca como alerta un WER (tasa de error por palabra) superior a `0.5`;
`--max-wer` permite cambiar ese umbral y `--asr-model` elegir otro modelo. La
transcripción reconocida y el WER quedan en el informe. Las alertas son una
ayuda para revisar, no una prueba concluyente: algunos subtítulos españoles
solo cubren parte del diálogo hablado. Puedes probar un subconjunto con
`--only EN_<clave>` o `--limit N`.
`--asr-all` controla qué audios se transcriben; `--report-all` controla qué
filas se escriben en el CSV.

Para transcribir todos los audios seleccionados, incluso los que no tengan
duración anómala, añade `--asr-all`:

```powershell
.\.venv\Scripts\python.exe .\review_spanish_audio.py --asr --asr-all --asr-device cuda --report .\audio_es_review_all_asr.csv
```

### 8. Reconstruir el paquete completo

Asegúrate de que `ffmpeg` funciona y ejecuta:

```powershell
.\.venv\Scripts\python.exe .\pack_tentacle.py
```

El empaquetador muestra el progreso y genera `source\tenta_es.cle` junto con
`source\tenta_es.cle.report.csv`. Sustituye todos los diálogos alemanes que
tengan un WAV español correspondiente; conserva los efectos alemanes y avisa
si alguna voz generada dura menos de la mitad que la original. Revisa esos
casos en el informe y escucha los WAV: una duración corta puede indicar que el
subtítulo solo cubría parte del diálogo, o que hubo algún problema al sintetizar ese audio.
También cambia en el paquete la etiqueta del audio de
«Alemán» a «Español», para que se visualice así en el menú de configuración del juego; el banco de audio seleccionado sigue siendo el mismo.

Si el archivo de salida o el informe ya existen, añade `--overwrite` para
reemplazarlos.

### 9. Instalar y probar en el juego

1. Cierra el juego.
2. Haz una copia de seguridad del `tenta.cle` original en la carpeta de
   instalación del juego.
3. Copia `source\tenta_es.cle` a esa carpeta y ponle el nombre `tenta.cle`.
4. Inicia el juego, selecciona «Español» en el menú de audio y comprueba varios
   diálogos.

Para volver al juego original, restaura la copia de seguridad. No reemplaces
`source\tenta.cle` por el paquete reconstruido: se necesita el original para
volver a ejecutar el flujo.
