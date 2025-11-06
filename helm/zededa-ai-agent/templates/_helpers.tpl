{{/*
Expand the name of the chart.
*/}}
{{- define "zededa-ai-agent.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "zededa-ai-agent.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "zededa-ai-agent.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "zededa-ai-agent.labels" -}}
helm.sh/chart: {{ include "zededa-ai-agent.chart" . }}
{{ include "zededa-ai-agent.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: zededa-ai-agent
{{- end }}

{{/*
Selector labels
*/}}
{{- define "zededa-ai-agent.selectorLabels" -}}
app.kubernetes.io/name: {{ include "zededa-ai-agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Web labels
*/}}
{{- define "zededa-ai-agent.web.labels" -}}
{{ include "zededa-ai-agent.labels" . }}
app.kubernetes.io/component: web
{{- end }}

{{/*
Web selector labels
*/}}
{{- define "zededa-ai-agent.web.selectorLabels" -}}
{{ include "zededa-ai-agent.selectorLabels" . }}
app.kubernetes.io/component: web
{{- end }}

{{/*
Ollama labels
*/}}
{{- define "zededa-ai-agent.ollama.labels" -}}
{{ include "zededa-ai-agent.labels" . }}
app.kubernetes.io/component: ollama
{{- end }}

{{/*
Ollama selector labels
*/}}
{{- define "zededa-ai-agent.ollama.selectorLabels" -}}
{{ include "zededa-ai-agent.selectorLabels" . }}
app.kubernetes.io/component: ollama
{{- end }}

{{/*
Resolve the Ollama base URL for the web service
*/}}
{{- define "zededa-ai-agent.ollamaUrl" -}}
{{- if .Values.web.ollama.url }}
{{- .Values.web.ollama.url }}
{{- else }}
{{- printf "http://%s-ollama:%d" (include "zededa-ai-agent.fullname" .) (.Values.ollama.service.port | int) }}
{{- end }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "zededa-ai-agent.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "zededa-ai-agent.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Get image pull secrets
*/}}
{{- define "zededa-ai-agent.imagePullSecrets" -}}
{{- if .Values.global.imagePullSecrets }}
{{- range .Values.global.imagePullSecrets }}
- name: {{ . }}
{{- end }}
{{- end }}
{{- end }}
