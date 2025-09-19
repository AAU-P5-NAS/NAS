from pydantic import BaseModel
from src.importer import DataImporter


class ExampleTypeStruct(BaseModel):
    example_field: str
    another_field: int
    yet_another_field: float


example_instance = ExampleTypeStruct(
    example_field="example", another_field=42, yet_another_field=3.14
)


def main():
    importer = DataImporter(filepath="src/az_images_data.csv")
    importer.show_random_sample(8)
    importer.show_random_sample(200000)
    importer.show_random_sample(678)


if __name__ == "__main__":
    main()
