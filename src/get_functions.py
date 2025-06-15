import argparse
import json
from inspect import getmembers, isclass, isfunction, ismodule, ismethod, isbuiltin
import functools


def get_functions(module, name):
    for member in getmembers(module, isfunction):
        print(name, ".", member)


module_set = set()


def recurse_module(name, framework):
    print(name)

    if framework == "torch":
        skip_list = [
            "torch._ops",
            "torch._classes",
            "torch.nn.quantized.modules.conv",
            "torch.nn.quantized.modules.functional_modules",
            "torch.nn.quantized.modules.utils",
        ]
        return name not in skip_list
    elif framework == "torch_xla":
        return True
    elif framework == "tensorflow":
        return True
    elif framework == "jax":
        return True



def include_module(name, framework, is_top_level):
    if framework == "torch":
        return not name.startswith("torch._")
    elif framework == "torch_xla":
        return not name.startswith("torch_xla._")
    elif framework == "tensorflow":
        return (
            (not is_top_level and name[0] != "_")
            or "tensorflow._api.v2" in name
            or "tensorboard.summary" in name
            or "tensorflow_estimator" in name
            or "keras" in name
            or "compat" in name
        )
    elif framework == "jax":
        return not name.startswith("jax._src")


def iterate_module(
    framework, module=None, is_top_level=True, namespace_dict={}, isClass=False
):
    # if module.__name__ == "torch.amp.autocast_mode":
    #     import code; code.interact(local=dict(globals(), **locals()))
    if is_top_level:
        namespace_dict['_version'] = f'{framework}-{module.__version__}'
    # else:
    #     if module.__name__ == "torch":
    #         import code; code.interact(local=dict(globals(), **locals()))
    for child in getmembers(module, isclass if isClass else ismodule):
        name = child[0]
        child = child[1]
        if child.__name__ not in module_set and (
            (
                include_module(child.__name__, framework, is_top_level)
                and module.__name__ in child.__name__
            )
            or isClass
        ):
            #print(f'child.__name__ is {child.__name__}')
            module_set.add(child.__name__)

            if framework == "tensorflow":
                key = child.__name__.removeprefix("tensorflow._api.v2.")
            else:
                key = child.__name__
            namespace_dict[key] = {
                "functions": [],
                "classes": {},
                "modules": {},
                "methods": [],
            }
            # print("child", child, hasattr(child, "__iter__"))
            # if child.__name__ =="torch.classes":
            #     import code; code.interact(local=dict(globals(), **locals()))
            # print(child)
            # if child.__name__ == "torch.distributions.lkj_cholesky":
            #     import code; code.interact(local=dict(globals(), **locals()))
            try:
                for member in getmembers(child, ismethod):
                    # print("fn", key, ".", member[0])
                    namespace_dict[key]["methods"].append(member[0])

                for member in getmembers(
                    child, lambda x: isfunction(x) or isbuiltin(x)
                ):
                    # print("fn", key, ".", member[0])
                    namespace_dict[key]["functions"].append(member[0])
                for member in getmembers(child, isclass):
                    # print("class", key, ".", member[0])
                    namespace_dict[key]["classes"][member[0]] = iterate_module(
                        framework,
                        child,
                        False,
                        {"functions": [], "classes": {}, "methods": []},
                        True,
                    )
                if not isClass:
                    namespace_dict[key]["modules"] = iterate_module(
                        framework, child, False, {}
                    )
            except TypeError:
                print("EXCLUDING", child)
                pass
        # elif is_top_level:
        #     print("EXCLUDING", child, child.__name__, child.__name__ not in module_set)
    return namespace_dict
    # if recurse_module(child.__name__, framework):

    # print("MODULE", child.__name__.removeprefix("tensorflow._api.v2."),
    #       child.__name__, name, module.__name__)



flat_set = set()


def top_count_functions(functions_dict):
    global flat_set
    function_count = 0
    class_count = 0
    for key, value in functions_dict.items():
        if key == '_version':
            continue
        if "modules" in value:
            (
                module_function_count,
                module_class_count,
                flat_set,
            ) = top_count_functions(value["modules"])
            function_count += module_function_count
            class_count += module_class_count
        class_function_count, class_class_count, flat_set = top_count_functions(
            value["classes"]
        )
        flat_set = flat_set.union(set(value["functions"]))
        flat_set = flat_set.union(set(value["classes"]))
        function_count += len(value["functions"]) + class_function_count
        class_count += len(value["classes"]) + class_class_count
    return function_count, class_count, flat_set


def write_funcs_to_file(framework, functions_dict, functions_set):
    f = open(functions_dict['_version'] + "_functions.json", "w")
    f.write(json.dumps(functions_dict, indent=4, sort_keys=True))
    f.close()
    f = open(functions_dict['_version'] + "_flat_functions.json", "w")
    f.write(json.dumps(list(functions_set), indent=4, sort_keys=True))
    f.close()


def get_framework_functions(framework):
    module = __import__(framework)
    
    function_dict = iterate_module(
        framework, module, is_top_level=True, namespace_dict={}
    )

    function_count, class_count , function_set  = top_count_functions(function_dict)
    print("FUNCTION COUNT", function_count, class_count)
    function_dict['function_count'] = function_count
    function_dict['class_count'] = class_count
    write_funcs_to_file(framework, function_dict, function_set)


def main():
    parser = argparse.ArgumentParser(description="A script that gets all functions from supported frameworks")
    
    parser.add_argument('-f', '--framework', action='store', help='specify the framwork: could be torch, torch_xla, jax, and tensorflow')

    args = parser.parse_args()

    supported_fw = ['jax', 'tensorflow', 'torch', 'torch_xla', 'xla']
    if args.framework in supported_fw:
        if args.framework == 'xla':
            args.framework = 'torch_xla'
        get_framework_functions(args.framework)
    else:
        print(parser.format_help()) 


if __name__ == "__main__":
    main()
