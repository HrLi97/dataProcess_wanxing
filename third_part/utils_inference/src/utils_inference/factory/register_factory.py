from typing import Dict, List, Callable


class Register(dict):
    def __init__(self, type: str = "default", *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._collection_type = type if type else "default"
        self._dict = {}  # type:Dict[str,Callable]

    def __call__(self, object_or_key: str | Callable) -> Callable:
        """
        generate function_wrap with key or function.__name

        :param func_or_key: key or function
        :type func_or_key: str | Callable
        :return: function itself, but set register[key]=function
        :rtype: Callable[..., Any]
        """
        if callable(object_or_key):
            return self.register(object_or_key)
        else:
            return lambda x: self.register(x, key=object_or_key)

    def register(self, callable_obj: Callable, key: str = None) -> Callable:
        if not callable(callable_obj):
            raise Exception(f"Error: {callable_obj} must be callable!")

        if key is None:
            key = callable_obj.__name__

        if key in self._dict:
            raise Exception(f"{key} already exists.")

        self[key] = callable_obj
        return callable_obj

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        if key not in self._dict:
            raise ValueError(
                f'register type:"{key}" not in Register, collection_type is "{self._collection_type}"'
            )
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str({k: v for k, v in self._dict.items()})

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def collection_type(self) -> str:
        return self._collection_type

    def object_types(self) -> str:
        return [str(key) for key in self.keys()]


class RegisterFactory(dict):
    def __init__(self, *args, **kwargs):
        super(RegisterFactory, self).__init__(*args, **kwargs)
        self._dict = {}  # type:Dict[str,Register]

    def make_register(self, type: str) -> Register:
        if type not in self._dict:
            self._dict[type] = Register(type=type)
        return self._dict[type]

    def __call__(
        self,
        object_type: str = None,
        collection_type: str = "default",
    ) -> Register:
        if object_type is None:
            return self.make_register(collection_type)
        else:
            return self.make_register(collection_type)(object_type)

    def set_register(self, value: Register):
        self[value.type()] = value

    def __setitem__(self, key, value):
        assert isinstance(value, Register), "value must be Register"
        self._dict[key] = value

    def __getitem__(self, key) -> Register:
        if key not in self._dict:
            raise ValueError(f'collection_type:"{key}" not in RegisterFactory')
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str({k: str(v) for k, v in self._dict.items()})

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def default_register(self) -> Register:
        return self.make_register("default")

    def collection_types(self) -> List[str]:
        return [str(key) for key in self.keys()]


if __name__ == "__main__":
    test_factory = RegisterFactory()

    @test_factory("cuda", "default")
    class TestCuda:
        def __init__(self):
            pass

        def test(self):
            print("test cuda...")

    @test_factory("cpu", "default")
    class TestCpu:
        def __init__(self):
            pass

        def test(self):
            print("test cpu...")

    @test_factory()
    class TestDefault:
        def __init__(self):
            pass

        def test(self):
            print("test default...")

    print(test_factory)

    test_factory.default_register()["cuda"]().test()
    test_factory.default_register()["TestDefault"]().test()
