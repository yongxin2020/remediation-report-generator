# Extracted Variables Reference

## Core Report Variables

### Session Metadata
| Variable | Type | Source | Description |
|----------|------|--------|-------------|
| `date_session_string` | String | log | Session date in DD-MM format (ex: "04 Mai") |
| `textual_start_time` | String | log | Human-readable session start time (ex: "11 heures et demie") |
| `nb_activités` | Integer | log | Total number of activities attempted |
| `nb_exercices` | Integer | log | Total number of exercises attempted |
| `duration_session_str` | String | log | Total session duration (ex: "environ une demi-heure") |

### Results
| Variable | Type | Source | Description |
|----------|------|--------|-------------|
| `num_raté` | Integer | log | Count of failed exercises |
| `num_moyen` | Integer | log | Count of partially successful activities |
| `taux_réussite` | Float | log | Overall success rate percentage (0-100) (ex: "50.0") |
| `exo_failed` | String | log | Names of failed exercises (ex: "L'exercice non réussi est : Exo 3 Que_d'accros (6ᵉ activité).") |

**Note**:
The first table in the report contains information about the exercises presented in THERADIA sessions. If you are using your own corpus or exercises, you will need to modify the relevant scripts.

### Linguistic Features
The following linguistic variables are extracted from CSV files and used in the final report (presented in the second table):

| Variable | Type | Source | Description |
|----------|-------------|--------|-----------------|
| `indicateur_higher` | List | transcript | Metrics significantly above population norm |
| `indicateur_lower` | List | transcript | Metrics significantly below population norm |
| `num_tokens_unique` | Integer | transcript | Count of distinct words used |
| `duration_utterance` | Float | transcript | Average utterance duration in seconds |
| `num_token_per_utterance` | Float | transcript | Average tokens per utterance |
| `num_output_pho` | Integer | transcript | Phonological output measure |
| `lexical_diversity_TTR` | Float | transcript | Type-Token Ratio (unique words/total words) |

These linguistic features are compared with the norm of the THERADIA corpus. For details on how the norm is calculated, see the *Norms Calculation* section below.

### Emotion
| Variable | Type | Source | Description |
|----------|------|--------|-------------|
| `salientEmotions_Positive` | Dict | audiovisual recordings | Significant positive emotions with intensities |
| `salientEmotions_Negative` | Dict | audiovisual recordings | Significant negative emotions with intensities |

For more information on how these variables are extracted and analyzed, see the *Emotion Analysis Pipeline* section.


