from od_base import ODBase
import json

class ObjectDetector(ODBase):
    _registry = {}

    @classmethod
    def register(cls, versions, model_names, tasks,frameworks,data={}):
        """
        Decorator to register a wrapper class for multiple versions, model names, and tasks.
        """
        def decorator(wrapper_cls):
            for framework in frameworks:
                for model_name in model_names:
                    for task in tasks:
                        for version in versions:
                            key = (framework, model_name, task, version,json.dumps(data))
                            if key in cls._registry:
                                raise ValueError(f"Wrapper already registered for version={version}, model_name='{model_name}', task='{task}'.")
                        cls._registry[key] = wrapper_cls
            return wrapper_cls
        return decorator
    
    def __new__(cls, version, model_name, task, framework,data={},*args, **kwargs):
        """
        Override __new__ to return the appropriate wrapper instance based on version, model_name, and task.
        """
        print(f"version={version}, model_name='{model_name}', task='{task}', framework='{framework}'")
        key = (framework, model_name, task, version, json.dumps(data))
        if key not in cls._registry:
            raise ValueError(f"Unsupported combination: version={version}, model_name='{model_name}', task='{task}'.")
        wrapper_cls = cls._registry[key]
        return wrapper_cls(*args, **kwargs)

