git submodule update --init --recursive
if errorlevel 1 exit /b %errorlevel%
git -C external/Atomic-Stockfish fetch --no-tags origin main:refs/remotes/origin/main
if errorlevel 1 exit /b %errorlevel%
cmake . -Bbuild -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX="./"
if errorlevel 1 exit /b %errorlevel%
cmake --build ./build --config RelWithDebInfo --target install
