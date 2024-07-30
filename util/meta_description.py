from itertools import product

import pandas as pd


def generate_meta_description(age=None, sex=None, loc=None, dev=None):
    # set up the prefix
    if loc or dev:
        output_sent = "This sound was recorded"
    else:
        output_sent = "This patient is"

    # prepare the main descriptions
    if loc:
        s_loc = "an unknown region" if loc == "unknown" else f"the {loc}"
        s_loc = f"from {s_loc}"

    if age and sex:
        if age == "unknown":
            s_age_sex = f"a {sex} patient of unknown age"
        elif sex == "unknown":
            s_age_sex = "an " if age == "adult" else "a "
            s_age_sex += f"{age} patient of unknown sex"
        elif age == "pediatric":
            s_age_sex = f"a {sex} {age} patient"
        else:
            s_age_sex = f"an {age} {sex} patient"
    elif age:
        if age == "unknown":
            s_age_sex = "a patient of unknown age"
        else:
            s_age_sex = "an " if age == "adult" else "a "
            s_age_sex += f"{age} patient"
    elif sex:
        if sex == "unknown":
            s_age_sex = "a patient of unknown sex"
        else:
            s_age_sex = f"a {sex} patient"

    if dev:
        if dev == "unknown":
            s_dev = "an unknown device"
        else:
            s_dev_prefix = "an" if dev == "AKGC417L" else "a"
            s_dev_suffix = "microphone" if dev == "AKGC417L" else "stethoscope"
            s_dev = f"{s_dev_prefix} {dev} {s_dev_suffix}"

    # combine the descriptions
    if "s_loc" in locals():
        output_sent += f" {s_loc}"
        if "s_age_sex" in locals():
            s_age_sex = s_age_sex.replace("of", "with")

    if "s_age_sex" in locals():
        if "s_loc" not in locals() and "s_dev" not in locals():
            output_sent += f" {s_age_sex}"
        elif "s_loc" in locals():
            output_sent += f" of {s_age_sex}"
        else:
            output_sent += f" from {s_age_sex}"

    if "s_age_sex" in locals() and "s_dev" in locals():
        output_sent += f", using {s_dev}"
    elif "s_dev" in locals():
        output_sent += f" with {s_dev}"

    output_sent += "."

    return output_sent


def generate_clap_meta_description(
    age=None, sex=None, loc=None, dev=None, sex_debiasing=True
):
    output_sent = ""

    # prep the descriptions about the patient (age & sex)
    if age and sex:
        if age == "unknown":
            s_patient = f"A {sex} patient of unknown age"
        elif sex == "unknown":
            s_patient = "An " if age == "adult" else "A "
            s_patient += f"{age} patient of unknown sex"
        elif age == "pediatric":
            s_patient = f"A {sex} {age} patient"
        else:
            s_patient = f"An {age} {sex} patient"
    elif age:
        if age == "unknown":
            s_patient = "A patient of unknown age"
        else:
            s_patient = "An " if age == "adult" else "A "
            s_patient += f"{age} patient"
    elif sex:
        if sex == "unknown":
            s_patient = "A patient of unknown sex"
        else:
            s_patient = f"A {sex} patient"
    else:
        s_patient = "A patient"

    # prep descriptions about the location
    if loc:
        s_loc = "an unknown region" if loc == "unknown" else f"the {loc}"
        s_loc = f"body sounds recorded from {s_loc}"
    else:
        s_loc = "body sounds recorded"

    if sex_debiasing:
        s_loc = f"their {s_loc}"
    else:
        if sex == "male":
            s_loc = f"his {s_loc}"
        elif sex == "female":
            s_loc = f"her {s_loc}"
        else:
            s_loc = f"their {s_loc}"

    # prep descriptions about the device
    if dev:
        if dev == "unknown":
            s_dev = "an unknown device"
        else:
            s_dev_prefix = "an" if dev == "AKGC417L" else "a"
            s_dev_suffix = "microphone" if dev == "AKGC417L" else "stethoscope"
            s_dev = f"{s_dev_prefix} {dev} {s_dev_suffix}"

    # combine the descriptions
    output_sent += f"{s_patient} had {s_loc}"

    if "s_dev" in locals():
        if loc:
            output_sent += f", with {s_dev}"
        else:
            output_sent += f" with {s_dev}"

    output_sent += "."

    return output_sent

'''
# set values for each variable
age_values = ["adult", "pediatric", "unknown", None]
sex_values = ["male", "female", "unknown", None]
loc_values = [
    "trachea",
    "left anterior chest",
    "right anterior chest",
    "left posterior chest",
    "right posterior chest",
    "left lateral chest",
    "right lateral chest",
    "unknown",
    None,
]
dev_values = ["Meditron", "LittC2SE", "Litt3200", "AKGC417L", "unknown", None]

# generate all possible combinations
meta_combinations = list(product(age_values, sex_values, loc_values, dev_values))
# remove combinations with more than 1 "unknown"
meta_combinations = [comb for comb in meta_combinations if comb.count("unknown") <= 1]
meta_combinations = [
    comb for comb in meta_combinations if comb != (None, None, None, None)
]

# # generate all meta descriptions
# meta_descriptions = []
# for comb in meta_combinations:
#     meta_dict = {"age": comb[0], "sex": comb[1], "loc": comb[2], "dev": comb[3]}
#     meta_description = generate_meta_description(**meta_dict)
#     meta_dict["meta_description"] = meta_description
#     meta_descriptions.append(meta_dict)

# meta_descriptions_df = pd.DataFrame(meta_descriptions)
# meta_descriptions_df.to_excel("meta_descriptions.xlsx", index=False)

# generate all meta descriptions
output_descriptions = []
for comb in meta_combinations:
    meta_dict = {"age": comb[0], "sex": comb[1], "loc": comb[2], "dev": comb[3]}
    output_dict = dict(meta_dict)

    # w/o sex debiasing (w/ "his", "her", "their")
    meta_description_wo_debiasing = generate_clap_meta_description(
        **meta_dict, sex_debiasing=False
    )
    output_dict["meta_description_wo_debiasing"] = meta_description_wo_debiasing

    # w/ sex debiasing (only w/ "their")
    meta_description = generate_clap_meta_description(**meta_dict)
    output_dict["meta_description"] = meta_description

    output_descriptions.append(output_dict)

output_descriptions_df = pd.DataFrame(output_descriptions)
output_descriptions_df.to_excel("meta_descriptions_clap_style.xlsx", index=False)
'''
