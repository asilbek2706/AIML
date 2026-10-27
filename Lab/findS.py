import csv

def find_s_algorithm(file_path):
    concepts = []
    target = []

    # CSV faylni ochish va o'qish
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)  # Sarlavha qatorini o'tkazib yuborish
        for row in reader:
            concepts.append(row[:-1])  # Atributlar (features)
            target.append(row[-1])  # Maqsadli ustun (EnjoySport: Yes/No)

    # 1-qadam: Boshlang'ich gipotezani aniqlash
    # Birinchi ijobiy (Yes) misolni boshlang'ich gipoteza sifatida olamiz
    hypothesis = None
    for i, t in enumerate(target):
        if t.strip().lower() == "yes":
            hypothesis = list(concepts[i])
            break

    if hypothesis is None:
        return "Datasetda ijobiy (Yes) misollar topilmadi!"

    print(f"Boshlang'ich xususiy gipoteza: {hypothesis}\n")

    # 2-qadam: Qolgan misollarni tekshirish va gipotezani yangilash
    for i, val in enumerate(target):
        if val.strip().lower() == "yes":
            # Har bir atributni joriy gipoteza bilan solishtiramiz
            for j in range(len(hypothesis)):
                if concepts[i][j] != hypothesis[j]:
                    hypothesis[j] = "?"  # Agar mos kelmasa, umumiyroq qilamiz

            print(f"{i + 1}-qator (Ijobiy): {concepts[i]} --> Gipoteza: {hypothesis}")
        else:
            print(
                f"{i + 1}-qator (Salbiy - O'tkazib yuborildi): {concepts[i]} -->"
                " O'zgarishsiz"
            )

    return hypothesis


# Funksiyani ishga tushirish (dataset.csv fayli bilan bir papkada bo'lishi kerak)
if __name__ == "__main__":
    file_name = "./data/dataset.csv"
    try:
        final_hypothesis = find_s_algorithm(file_name)
        print(f"\nYakuniy eng xususiy gipoteza: {final_hypothesis}")
    except FileNotFoundError:
        print(
            f"Xatolik: '{file_name}' fayli topilmadi. Iltimos, CSV faylni yarating."
        )
