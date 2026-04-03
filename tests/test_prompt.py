from voxlink.prompt import ASR, INSTRUCT, QA, PromptBuilder
from voxlink.vocab import SpecialTokens, build_joint_vocab


def _vocab():
    return build_joint_vocab(
        4, ["hello", "world"], task_tokens=[ASR.token(), QA.token(), INSTRUCT.token()]
    )


def test_prompt_assembly_order() -> None:
    vocab = _vocab()
    pb = PromptBuilder(vocab)
    ids = pb.build([0, 1, 2], ASR)
    tokens = vocab.decode(ids)
    assert tokens[0] == SpecialTokens.BOS
    assert tokens[1] == ASR.token()
    assert tokens[2:5] == ["u_0", "u_1", "u_2"]
    assert tokens[-1] == SpecialTokens.EOS


def test_qa_prompt_includes_text() -> None:
    vocab = _vocab()
    pb = PromptBuilder(vocab)
    ids = pb.build([0], QA, "hello world")
    tokens = vocab.decode(ids)
    assert "hello" in tokens and "world" in tokens


def test_qa_requires_text() -> None:
    pb = PromptBuilder(_vocab())
    try:
        pb.build([0], QA, "")
    except ValueError:
        return
    raise AssertionError("QA 缺少文本应抛 ValueError")


def test_decode_answer_skips_special_and_units() -> None:
    vocab = _vocab()
    pb = PromptBuilder(vocab)
    ids = vocab.encode([SpecialTokens.BOS, "u_1", "hello", "world", SpecialTokens.EOS])
    assert pb.decode_answer(ids) == "hello world"
