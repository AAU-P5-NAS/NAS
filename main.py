from src.importer import DataImporter


def main():
    print("This is the main function.")

    importer = DataImporter()

    ffnn_data = importer.get_as_ffnn()

    print("FFNN DataLoader:", ffnn_data)


if __name__ == "__main__":
    main()
