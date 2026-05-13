# Future Improvements — NX Computing AI

Registro de mejoras técnicas identificadas durante el desarrollo. Cada entrada documenta una posible implementación futura con suficiente contexto para evaluarla e implementarla sin tener que reconstruir la conversación original.

Ver regla 11 de CLAUDE.md para el formato de entradas y el protocolo completo.

---

## Resolución del tiler MJPEG configurable por cliente

**Descripción:** La resolución del preview MJPEG (nvmultistreamtiler) está hardcodeada a 1280×720. Sería útil exponerla en `config.yaml` como `tiler_width` / `tiler_height` para que instalaciones con menos cámaras puedan usar 1920×1080 y deployments con más restricciones de memoria puedan bajar más.

**Por qué sería mejor:** Flexibilidad sin cambiar código. Actualmente 1280×720 es un compromiso conservador; clientes con 4-6 cámaras podrían preferir preview en HD.

**Reemplazaría:**
- Archivo: `deploy/pipelines/app.py`
- Sección / función: construcción del tiler (líneas ~323-327)
- Descripción: valores hardcodeados `1280` y `720`

**Tech stack propuesto:**
- Leer `tiler_width` / `tiler_height` desde `config.yaml` vía `config_loader.py`, con defaults 1280/720

**Consideraciones:** Cambio menor. Cuidar que valores muy grandes no causen NVMM overflow en deployments de 16 cámaras (razón por la que se bajó de 1920×1080 a 1280×720).

---

<!-- Agregar entradas aquí siguiendo el formato:

## [Título de la mejora]

**Descripción:** qué es esta implementación futura y qué resuelve o mejora

**Por qué sería mejor:** ventaja concreta sobre la solución actual (precisión, velocidad, escalabilidad, etc.)

**Reemplazaría:**
- Archivo: `deploy/pipelines/probes.py`
- Sección / función: nombre de la función o clase (líneas aprox. XXX–XXX)
- Descripción de lo que se reemplaza

**Tech stack propuesto:**
- Modelo / librería: nombre + versión + licencia
- Forma de integración: SGIE / worker Python / reemplazo de config / etc.

**Consideraciones:** dependencias, tamaño del modelo, compatibilidad con Jetson Orin Nano, esfuerzo estimado

-->
