import json
import os
from pathlib import Path

CLAIMS_FILE = Path("tests/quality/test_emerging_threats.json")
ADVISORY_DIR = Path("data/synthetic_advisories")

CLAIMS_DATA = [
  {
    "claim": "Drinking ionized copper water completely cures the new XBB.1.5 variant.",
    "expected_label": "False",
    "topic": "XBB.1.5 Treatment",
    "source_file": "cdc_advisory_xbb15_copper.txt",
    "text": "CDC Health Advisory: XBB.1.5 Variant Treatment Guidelines.\nThe Centers for Disease Control and Prevention explicitly state there is no scientific evidence that drinking ionized copper water cures or prevents the XBB.1.5 variant. Copper toxicity can result from excessive consumption. Standard antiviral treatments remain the recommended protocol."
  },
  {
    "claim": "The synthetic food additive E-994 has been proven to reverse early-onset dementia.",
    "expected_label": "False",
    "topic": "Dementia Treatment",
    "source_file": "fda_advisory_e994.txt",
    "text": "FDA Safety Alert regarding E-994.\nThe synthetic food additive known as E-994 has NOT been proven to reverse, cure, or treat early-onset dementia. Furthermore, long-term ingestion of E-994 may lead to gastrointestinal distress. The FDA strongly advises against its use as a medical treatment."
  },
  {
    "claim": "Prolonged exposure to 6G cellular networks causes an immediate deterioration of the optic nerve.",
    "expected_label": "False",
    "topic": "6G Radiation",
    "source_file": "who_advisory_6g.txt",
    "text": "WHO Factsheet: 6G and Human Health.\nExtensive studies by the World Health Organization on 6G cellular networks confirm that exposure to these frequencies does not cause deterioration of the optic nerve. Claims linking 6G telecommunications infrastructure to immediate blindness or eye damage are unsubstantiated and false."
  },
  {
    "claim": "Applying peppermint essential oils to the soles of the feet prevents the transmission of monkeypox.",
    "expected_label": "False",
    "topic": "Mpox Prevention",
    "source_file": "cdc_advisory_mpox_oils.txt",
    "text": "CDC Guideline: Mpox Prevention Strategies.\nApplying peppermint or any other generic essential oils to the soles of the feet does not prevent the transmission of the Mpox virus. Mpox is primarily spread through close skin-to-skin contact. Vaccination and avoiding contact with infected individuals are the only confirmed preventive measures."
  },
  {
    "claim": "The newly discovered Sigma variant of influenza is entirely immune to all known antiviral medications.",
    "expected_label": "False",
    "topic": "Influenza Sigma Variant",
    "source_file": "who_advisory_sigma_flu.txt",
    "text": "WHO Update: The Sigma Variant of Influenza.\nThe recently identified Sigma variant of influenza remains susceptible to several classes of neuraminidase inhibitors, including oseltamivir. Reports claiming the Sigma variant is entirely immune to all known antiviral medications are categorically false."
  },
  {
    "claim": "Daily consumption of raw, unpasteurized camel milk can completely reverse Type 1 Diabetes.",
    "expected_label": "False",
    "topic": "Type 1 Diabetes",
    "source_file": "nih_advisory_camel_milk.txt",
    "text": "NIH Research Brief: Camel Milk and Diabetes.\nWhile camel milk contains insulin-like proteins, daily consumption of raw or pasteurized camel milk cannot \"completely reverse\" Type 1 Diabetes. Patients require exogenous insulin therapy. Suggesting camel milk as a replacement for insulin can lead to life-threatening diabetic ketoacidosis."
  },
  {
    "claim": "Intravenous injections of colloidal silver eliminate the Ebola virus from the bloodstream within 24 hours.",
    "expected_label": "False",
    "topic": "Ebola Virus",
    "source_file": "cdc_advisory_ebola_silver.txt",
    "text": "CDC Clinical Notice: Colloidal Silver and Ebola.\nIntravenous injection of colloidal silver is not an effective treatment for the Ebola virus and cannot eliminate it from the bloodstream. Furthermore, injecting colloidal silver can cause severe side effects including argyria and kidney failure. Approved monoclonal antibodies remain the standard of care."
  },
  {
    "claim": "The lab-grown meat alternative known as 'Neo-Beef' contains untested synthetic prions that cause brain damage.",
    "expected_label": "False",
    "topic": "Lab-Grown Meat Safety",
    "source_file": "fda_advisory_neo_beef.txt",
    "text": "FDA Safety Review: Lab-Grown 'Neo-Beef'.\nFollowing rigorous safety testing, the FDA has confirmed that the cellular agriculture product 'Neo-Beef' does not contain any synthetic prions, nor does it cause brain damage or transmissible spongiform encephalopathies. Social media rumors suggesting otherwise are baseless."
  },
  {
    "claim": "The 'Ozone 3' sleep therapy machines have been FDA approved for permanently curing chronic Lyme disease.",
    "expected_label": "False",
    "topic": "Chronic Lyme Disease",
    "source_file": "fda_advisory_ozone_lyme.txt",
    "text": "FDA Fraud Alert: Ozone 3 Therapy Machines.\nThe FDA has not approved 'Ozone 3' sleep therapy machines for diagnosing, treating, or permanently curing chronic Lyme disease. Ozone therapy involves significant health risks, particularly inhalation toxicity. Patients should consult licensed healthcare providers for Lyme disease management."
  },
  {
    "claim": "Micro-dosing lithium carbonate without a prescription is recommended for preventing post-COVID brain fog.",
    "expected_label": "False",
    "topic": "Post-COVID Symptoms",
    "source_file": "cdc_advisory_lithium_covid.txt",
    "text": "CDC Advisory: Post-COVID Conditions (Long COVID).\nThe CDC strongly advises against micro-dosing prescription medications such as lithium carbonate to prevent or treat post-COVID brain fog. Lithium has a narrow therapeutic index and unmonitored usage can cause severe toxicity and renal damage. There is no evidence supporting its efficacy for Long COVID."
  },
  {
    "claim": "Commercial mouthwashes containing chlorhexidine have been shown to temporarily alter human DNA.",
    "expected_label": "False",
    "topic": "Chlorhexidine Mouthwash",
    "source_file": "nih_advisory_chlorhexidine.txt",
    "text": "NIH Dental Health Summary: Chlorhexidine Safety.\nExtensive toxicological evaluations of chlorhexidine, a common antibacterial mouthwash ingredient, verify that it does not interact with, mutate, or alter human DNA. It is strictly an external antimicrobial agent. Claims of temporary DNA alteration are biologically impossible and inaccurate."
  },
  {
    "claim": "The 'Quantum-Infused' alkaline water brand is scientifically proven to regenerate damaged liver cells in cirrhosis patients.",
    "expected_label": "False",
    "topic": "Liver Cirrhosis",
    "source_file": "fda_advisory_quantum_water.txt",
    "text": "FDA Warning Letter: Quantum-Infused Alkaline Water.\nThe claims that 'Quantum-Infused' alkaline water can regenerate damaged liver cells in patients with cirrhosis are entirely false and scientifically unproven. Liver cirrhosis is typically irreversible. Marketing water as a cure for end-stage liver disease violates FDA regulations regarding medical claims."
  },
  {
    "claim": "Graphene oxide in municipal water supplies can be completely neutralized by adding a teaspoon of borax.",
    "expected_label": "False",
    "topic": "Water Supply Safety",
    "source_file": "epa_advisory_borax_water.txt",
    "text": "EPA Public Advisory: Municipal Water and Borax.\nMunicipal water supplies are not treated with graphene oxide. Furthermore, the EPA strongly warns against attempting to \"neutralize\" drinking water by adding borax (sodium tetraborate). Ingestion of borax can cause nausea, vomiting, and in larger doses, acute toxicity and organ failure."
  },
  {
    "claim": "Placing sliced raw onions around a sick person's bed draws the pneumonic plague bacteria out of the air.",
    "expected_label": "False",
    "topic": "Pneumonic Plague",
    "source_file": "who_advisory_onions_plague.txt",
    "text": "WHO Fact Sheet: Pneumonic Plague Treatment.\nThe pseudoscientific practice of placing sliced raw onions in a room does not attract, kill, or draw the pneumonic plague (Yersinia pestis) bacteria out of the air. Pneumonic plague is highly contagious and fatal without prompt antibiotic treatment within 24 hours."
  },
  {
    "claim": "Using neodymium magnets on vaccination injection sites will successfully realign blood iron and prevent mRNA integration.",
    "expected_label": "False",
    "topic": "Vaccine Safety",
    "source_file": "cdc_advisory_magnets_vaccines.txt",
    "text": "CDC Immunization Clarification: Myths about Magnets.\nApplying neodymium magnets to vaccination sites does absolutely nothing. The iron in human blood is not ferromagnetic, so magnets cannot \"realign\" it. Additionally, mRNA vaccines do not integrate into or alter human DNA. Both of these claims are medically and physically false."
  },
  {
    "claim": "Drinking tea made from boiled pine needles neutralizes the cardiovascular effects of vaccine spike proteins.",
    "expected_label": "False",
    "topic": "Spike Protein",
    "source_file": "nih_advisory_pine_needles.txt",
    "text": "NIH Botanical Safety Research.\nThere is no clinical evidence that drinking tea made from boiled pine needles neutralizes vaccine-derived spike proteins. While certain pine needles contain Vitamin C, some species (like Ponderosa pine) are toxic and can cause spontaneous abortion in mammals. It is neither safe nor effective for cardiovascular protection."
  },
  {
    "claim": "The artificial light emitted by new generation OLED screens is the primary cause of the newly classified 'Neo-Sars' virus.",
    "expected_label": "False",
    "topic": "Neo-Sars",
    "source_file": "who_advisory_neo_sars.txt",
    "text": "WHO Global Alert: The 'Neo-Sars' Hoax.\nThe World Health Organization confirms there is no such virus as 'Neo-Sars'. Furthermore, viruses are biological pathogens spread via droplets or aerosols; they are not caused or transmitted by the artificial light emitted from OLED screens or electronic devices. This claim is a complete fabrication."
  },
  {
    "claim": "Consuming Himalayan tar extract provides a 100% protective barrier against respiratory virus shedding from vaccinated individuals.",
    "expected_label": "False",
    "topic": "Virus Shedding",
    "source_file": "cdc_advisory_tar_extract.txt",
    "text": "CDC Fact Sheet: Vaccine Shedding Myths.\nIndividuals who receive mRNA or recombinant vaccines do not \"shed\" the virus, as these vaccines do not contain live virus. Therefore, there is no shedding to protect against. Consuming 'Himalayan tar extract' offers no respiratory protection and is not recognized by the FDA or CDC as safe for human consumption."
  },
  {
    "claim": "Deepfake-induced VR sickness is officially recognized as a neurological disorder curable only with mega-doses of pseudoephedrine.",
    "expected_label": "False",
    "topic": "VR Sickness",
    "source_file": "ama_advisory_vr_sickness.txt",
    "text": "AMA Position Statement on Virtual Reality Sickness.\nWhile extended virtual reality use can cause motion sickness, \"Deepfake-induced VR sickness\" is not a recognized neurological disorder. Crucially, taking mega-doses of pseudoephedrine is highly dangerous and can cause severe hypertension, arrhythmias, and seizures. It is not an approved treatment for VR sickness."
  },
  {
    "claim": "Titanium dioxide nanoparticles found in commercial sunscreens react dangerously when exposed to 5G cellular frequencies.",
    "expected_label": "False",
    "topic": "Sunscreen and 5G",
    "source_file": "fda_advisory_sunscreen_5g.txt",
    "text": "FDA Cosmetics Safety Update: Titanium Dioxide.\nTitanium dioxide is an FDA-approved, safe mineral UV filter used in sunscreens. Extensive testing demonstrates that these nanoparticles do not interact, combust, or react dangerously when exposed to 5G cellular network frequencies. The physics of radio waves do not cause photo-reactivity in mineral sunscreens."
  }
]


def main():
    os.makedirs(ADVISORY_DIR, exist_ok=True)
    os.makedirs(CLAIMS_FILE.parent, exist_ok=True)
    
    # Write json file
    claims_json = []
    for data in CLAIMS_DATA:
        claims_json.append({
            "claim": data["claim"],
            "expected_label": data["expected_label"],
            "topic": data["topic"],
            "source_file": data["source_file"]
        })
        
        # Write advisory txt file
        file_path = ADVISORY_DIR / data["source_file"]
        with open(file_path, "w") as f:
            f.write(data["text"])
            
    with open(CLAIMS_FILE, "w") as f:
        json.dump(claims_json, f, indent=2)

    print(f"Successfully generated {len(CLAIMS_DATA)} claims to {CLAIMS_FILE}")
    print(f"Successfully generated {len(CLAIMS_DATA)} advisory documents to {ADVISORY_DIR}")

if __name__ == "__main__":
    main()
