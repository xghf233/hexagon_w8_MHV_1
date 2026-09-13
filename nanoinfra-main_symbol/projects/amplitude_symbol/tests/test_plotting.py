"""Static-data helpers used by the report plotting script."""

from projects.amplitude_symbol.plot_word_bidi_report import uniform_sample_indices


def test_uniform_sampling_preserves_short_series_and_long_endpoints():
    short = uniform_sample_indices(4, max_points=10)
    long = uniform_sample_indices(1501, max_points=1000)

    assert short.tolist() == [0, 1, 2, 3]
    assert len(long) == 1000
    assert long[0] == 0
    assert long[-1] == 1500
    assert all(a < b for a, b in zip(long, long[1:]))
