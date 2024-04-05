from enum import Enum


class MoleculeNetTasks(Enum):
    tox21 = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
    clintox = [
        'FDA_APPROVED', 'CT_TOX'
    ]
    muv = [
            'MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
            'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
            'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859'
        ]
    sider = [
        'Hepatobiliary disorders', 'Metabolism and nutrition disorders',
        'Product issues', 'Eye disorders', 'Investigations',
        'Musculoskeletal and connective tissue disorders',
        'Gastrointestinal disorders', 'Social circumstances',
        'Immune system disorders', 'Reproductive system and breast disorders',
        'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
        'General disorders and administration site conditions',
        'Endocrine disorders', 'Surgical and medical procedures',
        'Vascular disorders', 'Blood and lymphatic system disorders',
        'Skin and subcutaneous tissue disorders',
        'Congenital, familial and genetic disorders', 'Infections and infestations',
        'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders',
        'Renal and urinary disorders',
        'Pregnancy, puerperium and perinatal conditions',
        'Ear and labyrinth disorders', 'Cardiac disorders',
        'Nervous system disorders', 'Injury, poisoning and procedural complications'
    ]
    hiv = [
        "HIV_active"
    ]