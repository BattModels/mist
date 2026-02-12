import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
from PIL import Image
from matplotlib import cm
from matplotlib.colors import Normalize


def get_color_mapper(scores):
    scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else scores
    vmin, vmax = scores_np.min(), scores_np.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.RdYlGn
    return norm, cmap


def map_tokens_to_structure(mol, tokens):
    """Map both atom and bond indices to token indices by parsing SMILES."""
    ALIPHATIC_ORGANIC = ["B", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    AROMATIC_ORGANIC = ["b", "c", "n", "o", "s", "p"]
    # fmt: off
    ELEMENT_SYMBOLS = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    ]
    # fmt: on
    BOND_SYMBOLS = {"-": 1, "=": 2, "#": 3, ":": 1.5, "/": 1, "\\": 1, ".": 0}
    SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>", "<unk>"]

    atom_symbols = set(ALIPHATIC_ORGANIC + AROMATIC_ORGANIC + ELEMENT_SYMBOLS)

    atom_map = {}
    bond_map = {}
    atom_count = 0
    branch_stack = []
    prev_atom = None
    pending_bond_token = None  # Track bond token waiting for next atom
    ring_closures = {}  # Track ring closure numbers: {ring_num: (atom_idx, token_idx)}
    in_bracket = False
    bracket_token_span = []  # Track all token indices in current bracket
    in_extended_ring = False  # Track '%' + digits rings
    extended_ring_tokens = []  # Track tokens for extended ring closure like ['%', '1', '0']

    for i, token in enumerate(tokens):
        if token in SPECIAL_TOKENS:
            continue

        # Handle bracketed atoms (e.g., [NH+] tokenized as ['[', 'N', 'H', '+', ']'])
        if token == "[":
            in_bracket = True
            bracket_token_span = [i]  # Start tracking bracket span
            continue
        elif token == "]" and in_bracket:
            in_bracket = False
            bracket_token_span.append(i)  # Include closing bracket
            # Complete the bracketed atom - map to all tokens in the bracket
            if atom_count < mol.GetNumAtoms():
                atom_map[atom_count] = bracket_token_span.copy()

                # Check for bond to previous atom
                if prev_atom is not None:
                    bond = mol.GetBondBetweenAtoms(prev_atom, atom_count)
                    if bond is not None:
                        # If there's an explicit bond token, use it; otherwise use bracket tokens for implicit bond
                        if pending_bond_token is not None:
                            bond_map[bond.GetIdx()] = [pending_bond_token]
                        else:
                            # Implicit bond - map to the bracket token span
                            bond_map[bond.GetIdx()] = bracket_token_span.copy()

                # Always clear pending_bond_token after processing an atom
                pending_bond_token = None
                prev_atom = atom_count
                atom_count += 1
            bracket_token_span = []
            continue
        elif in_bracket:
            # Track tokens inside brackets
            bracket_token_span.append(i)
            continue

        # Handle extended ring closures: %10 tokenized as ['%', '1', '0']
        if token == "%":
            in_extended_ring = True
            extended_ring_tokens = [i]  # Start with '%' token
            continue
        elif in_extended_ring and token.isdigit():
            extended_ring_tokens.append(i)
            continue
        elif in_extended_ring and not token.isdigit():
            # Process the ring closure with accumulated tokens
            ring_num = "%" + "".join(tokens[idx] for idx in extended_ring_tokens[1:])
            is_ring_closure = True
            ring_token_span = extended_ring_tokens
            in_extended_ring = False
            extended_ring_tokens = []
        else:
            is_ring_closure = token.isdigit()
            if is_ring_closure:
                ring_num = token
                ring_token_span = [i]

        is_atom = token in atom_symbols
        is_bond = token in BOND_SYMBOLS

        if is_atom and atom_count < mol.GetNumAtoms():
            atom_map[atom_count] = [i]  # Use list for consistency with bracketed atoms

            # Check for bond to previous atom
            if prev_atom is not None:
                bond = mol.GetBondBetweenAtoms(prev_atom, atom_count)
                if bond is not None:
                    # If there's an explicit bond token, use it; otherwise use current atom token for implicit bond
                    if pending_bond_token is not None:
                        bond_map[bond.GetIdx()] = [pending_bond_token]
                    else:
                        # Implicit bond - map to the current atom token
                        bond_map[bond.GetIdx()] = [i]

            # Always clear pending_bond_token after processing an atom
            pending_bond_token = None
            prev_atom = atom_count
            atom_count += 1
        elif is_bond:
            # Store the bond token to map when we see the next atom
            pending_bond_token = i
        elif is_ring_closure and prev_atom is not None:
            # Handle ring closures (e.g., '1', '2', '%10')
            # Check if there's a bond symbol before this ring closure (e.g., =1 or C=1)
            has_explicit_bond = pending_bond_token is not None
            # Use the explicit bond token if present, otherwise use the ring token span
            bond_token_indices = (
                [pending_bond_token] if has_explicit_bond else ring_token_span
            )
            pending_bond_token = None  # Clear after using

            if ring_num in ring_closures:
                # Second occurrence: close the ring
                first_atom, first_bond_token_indices, first_has_explicit = (
                    ring_closures[ring_num]
                )
                bond = mol.GetBondBetweenAtoms(first_atom, prev_atom)
                if bond is not None:
                    # Prefer explicit bond symbols over digit tokens
                    # Use whichever occurrence has an explicit bond symbol
                    if has_explicit_bond or first_has_explicit:
                        # Use the one with explicit bond
                        bond_map[bond.GetIdx()] = (
                            bond_token_indices
                            if has_explicit_bond
                            else first_bond_token_indices
                        )
                    else:
                        # Neither has explicit bond, use first occurrence digit(s)
                        bond_map[bond.GetIdx()] = first_bond_token_indices
                del ring_closures[ring_num]
            else:
                # First occurrence: store it with its bond token indices and whether it's explicit
                ring_closures[ring_num] = (
                    prev_atom,
                    bond_token_indices,
                    has_explicit_bond,
                )
        elif token == "(":
            # Push current atom onto stack for branch
            if prev_atom is not None:
                branch_stack.append(prev_atom)
        elif token == ")":
            # Pop from stack to return to main chain
            if branch_stack:
                prev_atom = branch_stack.pop()
                pending_bond_token = None

    # Handle case where extended ring closure is at the end
    if in_extended_ring and extended_ring_tokens and prev_atom is not None:
        ring_num = "%" + "".join(tokens[idx] for idx in extended_ring_tokens[1:])
        ring_token_span = extended_ring_tokens
        has_explicit_bond = (
            False  # Can't have explicit bond if we're still collecting digits
        )
        bond_token_indices = ring_token_span

        if ring_num in ring_closures:
            first_atom, first_bond_token_indices, first_has_explicit = ring_closures[
                ring_num
            ]
            bond = mol.GetBondBetweenAtoms(first_atom, prev_atom)
            if bond is not None:
                bond_map[bond.GetIdx()] = (
                    first_bond_token_indices
                    if first_has_explicit
                    else bond_token_indices
                )
        else:
            ring_closures[ring_num] = (prev_atom, bond_token_indices, has_explicit_bond)

    return atom_map, bond_map


def draw_molecule_with_attributions(
    smiles, tokens, attribution_scores, norm=None, cmap=None
):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if not mol:
        return None

    AllChem.Compute2DCoords(mol)
    scores_np = (
        attribution_scores.cpu().numpy()
        if torch.is_tensor(attribution_scores)
        else attribution_scores
    )

    # Use provided norm/cmap if available, otherwise create local one
    if norm is None or cmap is None:
        norm, cmap = get_color_mapper(attribution_scores)

    # Map atoms and bonds to their corresponding token indices
    atom_to_token, bond_to_token = map_tokens_to_structure(mol, tokens)

    atom_colors = {}
    for atom_idx, token_indices in atom_to_token.items():
        # Aggregate scores across all tokens for this atom (average)
        valid_indices = [idx for idx in token_indices if idx < len(scores_np)]
        if valid_indices:
            aggregated_score = np.mean([scores_np[idx] for idx in valid_indices])
            color_val = cmap(norm(aggregated_score))
            atom_colors[atom_idx] = color_val[:3]

    bond_colors = {}
    for bond_idx, token_indices in bond_to_token.items():
        # Aggregate scores across all tokens for this bond (average)
        valid_indices = [idx for idx in token_indices if idx < len(scores_np)]
        if valid_indices:
            aggregated_score = np.mean([scores_np[idx] for idx in valid_indices])
            color_val = cmap(norm(aggregated_score))
            bond_colors[bond_idx] = color_val[:3]

    drawer = rdMolDraw2D.MolDraw2DCairo(1200, 1200)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightBonds=list(bond_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    img_bytes = drawer.GetDrawingText()
    return Image.open(BytesIO(img_bytes))
