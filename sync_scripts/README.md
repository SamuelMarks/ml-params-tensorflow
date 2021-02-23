sync_scripts
============

For use in CI/CD, it helps to centralise all operations in one place.

Here is a nice a dynamic example for what cdd allows you to do:



It can be a bit confronting for people new to this project, so to lay those issues, here's the birdseye, with examples:

## `sync_properties`

Say you have a mapping like:

    { "foo": ("can", "haz") }

…and a class like:

    class Foo(object):
        foo = "haz"

Then `python -m cdd sync_properties` will get you:

    class Foo(object):
        foo: Literal["can", "haz"] = "haz"

## `sync`

For examples of sync see cdd project.
