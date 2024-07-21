from itertools import product

from mendeleev import Element, get_all_elements


def all_elements():
    for e in get_all_elements():
        return e.symbol


def enumerate_element(x: Element, include_aromatic=True):
    symbol = x.symbol
    if symbol in ["B", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]:
        yield symbol

    if symbol in ["B", "C", "N", "O", "P", "S"] and include_aromatic:
        yield symbol.lower()
        symbol = [symbol, symbol.lower()]
    else:
        symbol = [symbol]

    grid = product(
        symbol,
        x.oxistates,
        (i.mass_number for i in x.isotopes),
        ["", "H"],
        ["", "@", "@@"],
    )
    for symbol, charge, isotope, hydrogen, chiral in grid:
        yield "[" + str(isotope) + symbol + chiral + hydrogen + f"{charge:+d}" + "]"


def all_bracketed_tokens(**kwargs):
    for e in get_all_elements():
        for tok in enumerate_element(e, **kwargs):
            yield tok
