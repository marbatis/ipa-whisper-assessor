from ipa_whisper_assessor.align import PredWord, align_words


def test_align_substitution_z_to_s():
    ref_words = ["zoo"]
    expected = ["zuː"]
    predicted = [PredWord(ipa="suː", start=0.0, end=0.5)]
    aligned = align_words(ref_words, expected, predicted)
    assert len(aligned) == 1
    ops = aligned[0].phoneme_ops
    assert any(o.op == "sub" and o.expected == "z" and o.predicted == "s" for o in ops)

