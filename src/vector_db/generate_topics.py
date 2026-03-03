import os

# Configuration
OUTPUT_FILE = "data/topics.txt"

# 1. GENERAL & PUBLIC HEALTH (50)
public_health = [
    "Public health policy", "Epidemiology basics", "Global health initiatives", "Pandemic preparedness",
    "Vaccination schedules", "Herd immunity", "Antibiotic resistance", "Health equity",
    "Social determinants of health", "Telemedicine adoption", "Electronic health records",
    "Patient data privacy HIPAA", "Medical ethics", "Palliative care", "Hospice care",
    "Health insurance systems", "Occupational safety health", "Environmental health",
    "Air pollution health effects", "Water quality and health", "Food safety guidelines",
    "Maternal mortality rates", "Infant mortality reduction", "Life expectancy trends",
    "Aging population challenges", "Healthcare accessibility", "Rural healthcare",
    "Urban health issues", "Infectious disease control", "Chronic disease management",
    "Preventive medicine", "Alternative medicine", "Integrative health", "Holistic medicine",
    "Placebo effect", "Clinical trial phases", "Evidence based medicine", "Peer review process",
    "Medical misinformation", "Health literacy", "Doctor patient communication",
    "Hospital infection control", "Emergency room triage", "First aid basics", "CPR guidelines",
    "Blood donation safety", "Organ donation ethics", "Stem cell research", "Genetic testing ethics"
]

# 2. COMMON DISEASES & CONDITIONS (150)
diseases = [
    "Type 1 Diabetes", "Type 2 Diabetes", "Gestational Diabetes", "Hypertension", "Hypotension",
    "Coronary artery disease", "Heart failure", "Arrhythmia", "Stroke", "Atherosclerosis",
    "Deep vein thrombosis", "Varicose veins", "Anemia", "Hemophilia", "Leukemia", "Lymphoma",
    "Asthma", "COPD", "Bronchitis", "Pneumonia", "Tuberculosis", "Lung cancer", "Cystic fibrosis",
    "Sleep apnea", "Common cold", "Influenza", "COVID-19", "Measles", "Mumps", "Rubella",
    "Chickenpox", "Shingles", "Hepatitis A", "Hepatitis B", "Hepatitis C", "Cirrhosis",
    "Fatty liver disease", "Gallstones", "Kidney stones", "Chronic kidney disease",
    "Urinary tract infection", "Prostate cancer", "Breast cancer", "Ovarian cancer",
    "Cervical cancer", "Endometriosis", "Polycystic ovary syndrome", "Menopause",
    "Erectile dysfunction", "Male infertility", "Female infertility", "HIV AIDS", "Gonorrhea",
    "Syphilis", "Chlamydia", "Herpes simplex", "HPV infection", "Lyme disease", "Malaria",
    "Dengue fever", "Zika virus", "Ebola virus", "Rabies", "Tetanus", "Cholera", "Typhoid fever",
    "Food poisoning", "Salmonella", "E. coli infection", "Peptic ulcer", "GERD", "IBS",
    "Crohns disease", "Ulcerative colitis", "Celiac disease", "Lactose intolerance",
    "Appendicitis", "Hemorrhoids", "Arthritis", "Osteoarthritis", "Rheumatoid arthritis",
    "Osteoporosis", "Gout", "Fibromyalgia", "Lupus", "Multiple sclerosis", "ALS",
    "Parkinsons disease", "Alzheimers disease", "Dementia", "Epilepsy", "Migraine",
    "Cluster headaches", "Concussion", "Traumatic brain injury", "Spinal cord injury",
    "Sciatica", "Carpal tunnel syndrome", "Tendinitis", "Plantar fasciitis", "Eczema",
    "Psoriasis", "Acne vulgaris", "Rosacea", "Melanoma", "Basal cell carcinoma",
    "Hives", "Ringworm", "Athletes foot", "Glaucoma", "Cataracts", "Macular degeneration",
    "Conjunctivitis", "Hearing loss", "Tinnitus", "Vertigo", "Sinusitis", "Tonsillitis",
    "Hypothyroidism", "Hyperthyroidism", "Hashimotos thyroiditis", "Graves disease",
    "Addisons disease", "Cushings syndrome", "Type 1 allergies", "Seasonal allergies",
    "Food allergies", "Anaphylaxis", "Autoimmune diseases", "Rare diseases", "Genetic disorders",
    "Down syndrome", "Autism spectrum disorder", "ADHD"
]

# 3. MENTAL HEALTH (50)
mental_health = [
    "Generalized anxiety disorder", "Panic disorder", "Social anxiety", "Major depressive disorder",
    "Bipolar disorder", "Schizophrenia", "OCD", "PTSD", "Eating disorders", "Anorexia nervosa",
    "Bulimia nervosa", "Binge eating disorder", "Substance abuse", "Alcoholism", "Opioid addiction",
    "Nicotine dependence", "Gambling addiction", "Insomnia", "Narcolepsy", "Sleep paralysis",
    "Stress management", "Burnout syndrome", "Mindfulness meditation", "Cognitive behavioral therapy",
    "Psychotherapy", "Psychiatry medications", "Antidepressants", "Antipsychotics", "Mood stabilizers",
    "Suicide prevention", "Self harm", "Grief counseling", "Postpartum depression",
    "Seasonal affective disorder", "Personality disorders", "Borderline personality disorder",
    "Narcissistic personality disorder", "Antisocial personality disorder", "Dissociative disorders",
    "Child psychology", "Adolescent mental health", "Geriatric psychiatry", "Mental health stigma",
    "Digital detox", "Emotional intelligence", "Resilience building", "Trauma informed care"
]

# 4. NUTRITION & LIFESTYLE (100)
nutrition = [
    "Macronutrients", "Micronutrients", "Vitamins and minerals", "Vitamin D deficiency",
    "Iron deficiency", "Calcium requirements", "Protein intake", "Carbohydrates types",
    "Healthy fats", "Trans fats risks", "Sugar consumption", "Artificial sweeteners",
    "Hydration importance", "Caffeine effects", "Alcohol guidelines", "Mediterranean diet",
    "Keto diet", "Paleo diet", "Vegan diet", "Vegetarian diet", "Gluten free diet",
    "Intermittent fasting", "Calorie counting", "BMI accuracy", "Body composition",
    "Metabolism basics", "Probiotics and gut health", "Prebiotics", "Fiber intake",
    "Antioxidants", "Superfoods", "Processed foods", "Organic foods", "GMO foods safety",
    "Food labeling", "Nutritional supplements", "Multivitamins", "Fish oil benefits",
    "Creatine", "Whey protein", "Plant based protein", "Electrolytes", "Sports nutrition",
    "Meal planning", "Mindful eating", "Obesity prevention", "Weight loss strategies",
    "Weight gain strategies", "Sedentary lifestyle risks", "Aerobic exercise",
    "Anaerobic exercise", "HIIT training", "Strength training", "Yoga benefits",
    "Pilates benefits", "Cardiovascular fitness", "Flexibility training", "Balance exercises",
    "Exercise for seniors", "Exercise for pregnancy", "Rest and recovery", "Overtraining syndrome",
    "Sleep hygiene", "Circadian rhythms", "Blue light effects", "Screen time health",
    "Ergonomics", "Posture correction", "Smoking cessation", "Vaping health risks",
    "Sun safety", "Skin cancer prevention", "Oral hygiene", "Dental cavities",
    "Gum disease", "Sexual health", "Contraception methods", "Safe sex practices",
    "Pregnancy nutrition", "Breastfeeding benefits", "Childhood obesity", "School lunch nutrition"
]

# 5. TREATMENTS & PROCEDURES (100)
treatments = [
    "Antibiotics classes", "Antivirals", "Antifungals", "Pain management", "NSAIDs side effects",
    "Opioids risks", "Anesthesia types", "Local anesthesia", "General anesthesia",
    "Surgery preparation", "Post operative care", "Laparoscopic surgery", "Robotic surgery",
    "Organ transplant", "Blood transfusion", "Dialysis", "Chemotherapy", "Radiation therapy",
    "Immunotherapy", "Physical therapy", "Occupational therapy", "Speech therapy",
    "Chiropractic care", "Acupuncture", "Massage therapy", "Cognitive rehabilitation",
    "Vaccine types", "mRNA vaccines", "Flu shot", "Tetanus shot", "HPV vaccine",
    "X ray imaging", "MRI scan", "CT scan", "Ultrasound", "PET scan", "Mammography",
    "Colonoscopy", "Endoscopy", "Biopsy", "Blood tests interpretation", "Urinalysis",
    "Genetic screening", "Prenatal testing", "Amniocentesis", "IVF procedure",
    "Cesarean section", "Natural birth", "Vasectomy", "Tubal ligation", "Hysterectomy",
    "Mastectomy", "Appendectomy", "Tonsillectomy", "Cataract surgery", "LASIK surgery",
    "Dental implants", "Root canal", "Braces and orthodontics", "Teeth whitening",
    "Hearing aids", "Cochlear implants", "Pacemakers", "Stents", "Bypass surgery",
    "Angioplasty", "Valve replacement", "Joint replacement", "Knee replacement",
    "Hip replacement", "Arthroscopy", "Spinal fusion", "Insulin therapy", "Metformin",
    "Statins", "Beta blockers", "ACE inhibitors", "Diuretics", "Blood thinners",
    "Anticoagulants", "Immunosuppressants", "Corticosteroids", "Inhalers", "Epipen",
    "Antihistamines", "Proton pump inhibitors", "Antacids", "Laxatives",
    "Hormone replacement therapy", "Contraceptive pills", "IUD"
]

# 6. ANATOMY & PHYSIOLOGY (50)
anatomy = [
    "Cardiovascular system", "Respiratory system", "Digestive system", "Nervous system",
    "Endocrine system", "Immune system", "Lymphatic system", "Muscular system",
    "Skeletal system", "Reproductive system", "Urinary system", "Integumentary system",
    "Human brain structure", "Heart anatomy", "Lung function", "Liver function",
    "Kidney function", "Stomach anatomy", "Intestines function", "Pancreas function",
    "Thyroid gland", "Adrenal glands", "Pituitary gland", "Pineal gland",
    "Blood composition", "Red blood cells", "White blood cells", "Platelets",
    "Plasma", "DNA structure", "RNA function", "Gene expression", "Chromosomes",
    "Cell division mitosis", "Cell division meiosis", "Enzymes function", "Hormones types",
    "Neurotransmitters", "Dopamine", "Serotonin", "Cortisol", "Adrenaline",
    "Insulin regulation", "Blood pressure regulation", "Body temperature regulation",
    "Digestion process", "Immune response", "Inflammation mechanism", "Wound healing process"
]

def main():
    # Combine all lists
    all_topics = (
        public_health +
        diseases +
        mental_health +
        nutrition +
        treatments +
        anatomy
    )
    
    # Remove duplicates and sort
    unique_topics = sorted(list(set(all_topics)))
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for topic in unique_topics:
            f.write(f"{topic}\n")
            
    print(f"Successfully generated {len(unique_topics)} topics in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()