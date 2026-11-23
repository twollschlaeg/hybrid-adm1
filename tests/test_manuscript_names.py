"""The names the manuscript gives the datasets and reactors.

The data keep the laboratory's names (``LaRA-temp_F5``); the manuscript calls the
campaigns datasets A to C and a reactor by its dataset's letter and its number in it.
"""

from __future__ import annotations

import manuscript_figures
from helpers import canonical_split


def test_every_reactor_of_the_split_gets_its_own_name():
    names = sorted(manuscript_figures.name_of(label) for label in canonical_split().all_labels)
    assert names == ["A1", "A2", "A3", *(f"B{i}" for i in range(1, 7)),
                     *(f"C{i}" for i in range(1, 7))]


def test_the_names_keep_the_laboratory_numbers():
    assert manuscript_figures.name_of("rice-bran_R30") == "A1"
    assert manuscript_figures.name_of("rice-bran_R60") == "A2"
    assert manuscript_figures.name_of("rice-bran_R90") == "A3"
    assert manuscript_figures.name_of("LaRA-temp_F5") == "B5"
    assert manuscript_figures.name_of("LaRA-substrate_F4") == "C4"


def test_a_panel_is_titled_by_the_reactor_name():
    assert manuscript_figures.title_of("LaRA-temp_F5") == "Reactor B5"
