{{- define "inferflux.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "inferflux.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "inferflux.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
