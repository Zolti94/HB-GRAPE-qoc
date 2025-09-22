import compileall
ok = compileall.compile_dir('src', force=True, quiet=1)
print('compileall ok:', ok)
