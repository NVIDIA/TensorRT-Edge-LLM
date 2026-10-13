# Inja

This directory vendors the single-header distribution of
[Pantor Inja](https://github.com/pantor/inja) 3.5.0 at commit
`7d1b4600b68595085a949743331c2e5673f511ea`.

`include/inja/inja.hpp` extends that revision with the Jinja syntax and runtime
semantics used by supported provider chat templates. The extensions cover
macros, namespaces, collection literals and filters, loop controls, slicing,
undefined values, Jinja tests, and Python-compatible rendering. Keep changes
to the vendored header isolated in this directory and verify them through
`unittests/cpp/chatTemplate/chatTemplateTests.cpp`.
