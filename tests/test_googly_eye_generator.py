from eyegoogler.model.googly_eye_generator import GooglyEyeGenerator, GooglyEyeConfig


def test_eye_generator():
    size_scale = 0.5
    gen = GooglyEyeGenerator(GooglyEyeConfig(size_scale=size_scale))
    actual = gen.generate(100)
    assert 100 * 0.75 * size_scale <= actual.shape[0] <= 100 * 1.25 * size_scale
    assert 100 * 0.75 * size_scale <= actual.shape[1] <= 100 * 1.25 * size_scale
