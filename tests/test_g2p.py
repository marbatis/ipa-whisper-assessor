from ipa_whisper_assessor.g2p import G2POptions, g2p_words


def test_g2p_cmudict_non_empty_for_common_word():
    out = g2p_words(["zoo"], G2POptions(backend="cmudict"))
    assert out[0] != ""

