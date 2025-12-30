from ipa_whisper_assessor.ipa_tokenize import tokenize_ipa


def test_tokenize_affricate_tie_bar():
    assert tokenize_ipa("t͡ʃ") == ["t͡ʃ"]


def test_tokenize_diphthong():
    assert tokenize_ipa("aɪ") == ["aɪ"]


def test_tokenize_stress_and_length():
    assert tokenize_ipa("ˈziː") == ["ˈ", "z", "iː"]

