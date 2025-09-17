from pydantic import BaseModel


class ExampleTypeStruct(BaseModel):
    example_field: str
    another_field: int
    yet_another_field: float


example_instance = ExampleTypeStruct(
    example_field="example", another_field=42, yet_another_field=3.14
)


def main():
    print("example_instance", example_instance)
    print("example_instance.json()", example_instance.model_dump())
    print("example_field", example_instance.example_field)
    print("another_field", example_instance.another_field)
    print("yet_another_field", example_instance.yet_another_field)
    print("type of example_instance.example_field", type(example_instance.example_field))
    print("type of example_instance.another_field", type(example_instance.another_field))
    print(
        "type of example_instance.yet_another_field",
        type(example_instance.yet_another_field),
    )


if __name__ == "__main__":
    main()
