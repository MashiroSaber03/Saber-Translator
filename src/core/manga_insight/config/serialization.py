"""
Manga Insight 配置序列化工具

提供通用的 dataclass 序列化/反序列化 Mixin，消除重复代码。
"""

from dataclasses import dataclass, fields, field, MISSING
from typing import TypeVar, Type, Dict, Any, get_type_hints, get_origin, get_args, Union
import logging

logger = logging.getLogger("MangaInsight.Config")

T = TypeVar('T')


class SerializableMixin:
    """
    可序列化 Mixin

    为 dataclass 提供自动的 to_dict/from_dict 方法，
    支持嵌套 dataclass、可选字段、默认值等。

    使用方式:
        @dataclass
        class MyConfig(SerializableMixin):
            name: str = "default"
            count: int = 0

        config = MyConfig.from_dict({"name": "test"})
        data = config.to_dict()
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        将对象序列化为字典

        自动处理:
        - 基础类型直接转换
        - 嵌套的 SerializableMixin 对象递归调用 to_dict
        - 列表和字典中的嵌套对象
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            result[f.name] = self._serialize_value(value)
        return result

    def _serialize_value(self, value: Any) -> Any:
        """序列化单个值"""
        if value is None:
            return None
        if isinstance(value, SerializableMixin):
            return value.to_dict()
        if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
            return value.to_dict()
        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        # 基础类型直接返回
        return value

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        从字典反序列化为对象

        自动处理:
        - 缺失字段使用默认值
        - 嵌套的 SerializableMixin 类递归调用 from_dict
        - 类型转换和验证
        """
        if data is None:
            data = {}

        kwargs = {}
        type_hints = get_type_hints(cls) if hasattr(cls, '__dataclass_fields__') else {}

        for f in fields(cls):
            field_name = f.name
            field_type = type_hints.get(field_name, f.type)

            if field_name in data:
                raw_value = data[field_name]
                kwargs[field_name] = cls._deserialize_value(raw_value, field_type)
            # 如果字段不在数据中，使用默认值（由 dataclass 自动处理）

        return cls(**kwargs)

    @classmethod
    def _deserialize_value(cls, value: Any, field_type: Any) -> Any:
        """反序列化单个值"""
        if value is None:
            return None

        # 处理 Optional[X] 类型
        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            # Optional[X] 实际上是 Union[X, None]
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                field_type = non_none_args[0]
                origin = get_origin(field_type)

        # 处理 List[X] 类型
        if origin is list:
            args = get_args(field_type)
            if args:
                item_type = args[0]
                return [cls._deserialize_value(item, item_type) for item in value]
            return value

        # 处理 Dict[K, V] 类型
        if origin is dict:
            args = get_args(field_type)
            if len(args) == 2:
                value_type = args[1]
                return {k: cls._deserialize_value(v, value_type) for k, v in value.items()}
            return value

        # 处理嵌套的 SerializableMixin 类
        if isinstance(field_type, type) and issubclass(field_type, SerializableMixin):
            if isinstance(value, dict):
                return field_type.from_dict(value)
            return value

        if isinstance(field_type, type) and hasattr(field_type, "from_dict") and callable(getattr(field_type, "from_dict")):
            if isinstance(value, dict):
                return field_type.from_dict(value)
            return value

        # 基础类型直接返回
        return value


def create_default_factory(cls: Type[T]):
    """
    创建默认工厂函数

    用于 dataclass 的 field(default_factory=...)
    """
    def factory() -> T:
        return cls()
    return factory
