// LanguageConfig.kt
// Copy of the SDK LanguageRegistry.
// Add a new language here + in config.py. Nothing else changes.

package com.indicai.sdk

data class LanguageConfig(
    val code:        String,
    val name:        String,
    val whisperLang: String,
    val flores:      String,
    val sttPrompt:   String,
    val hasTts:      Boolean = true,
)

object LanguageRegistry {
    val SUPPORTED_LANGUAGES: Map<String, LanguageConfig> = mapOf(
        "hi" to LanguageConfig("hi", "Hindi",     "hi", "hin_Deva", "हिंदी में लिखें।"),
        "mr" to LanguageConfig("mr", "Marathi",   "mr", "mar_Deva", "मराठी मध्ये लिहा।"),
        "ta" to LanguageConfig("ta", "Tamil",     "ta", "tam_Taml", "தமிழில் எழுதுக।"),
        "te" to LanguageConfig("te", "Telugu",    "te", "tel_Telu", "తెలుగులో రాయండి।"),
        "kn" to LanguageConfig("kn", "Kannada",   "kn", "kan_Knda", "ಕನ್ನಡದಲ್ಲಿ ಬರೆಯಿರಿ।"),
        "ml" to LanguageConfig("ml", "Malayalam", "ml", "mal_Mlym", "മലയാളത്തിൽ എഴുതുക।"),
        "bn" to LanguageConfig("bn", "Bengali",   "bn", "ben_Beng", "বাংলায় লিখুন।"),
        "gu" to LanguageConfig("gu", "Gujarati",  "gu", "guj_Gujr", "ગુજરાતીમાં લખો।"),
        "pa" to LanguageConfig("pa", "Punjabi",   "pa", "pan_Guru", "ਪੰਜਾਬੀ ਵਿੱਚ ਲਿਖੋ।"),
        "ur" to LanguageConfig("ur", "Urdu",      "ur", "urd_Arab", "اردو میں لکھیں۔"),
    )

    fun get(code: String) = SUPPORTED_LANGUAGES[code]
        ?: throw IllegalArgumentException("Language '$code' not supported. Supported: ${SUPPORTED_LANGUAGES.keys}")

    fun isSupported(code: String) = code in SUPPORTED_LANGUAGES
}
