cmake_minimum_required(VERSION 3.22)

#
# This file is generated only once,
# and is not re-generated if converter is called multiple times.
#
# User is free to modify the file as much as necessary
#

# Setup compiler settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)


# Define the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# Set the project name
set(CMAKE_PROJECT_NAME main1)

# Enable compile command to ease indexing with e.g. clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Core project settings
project(${CMAKE_PROJECT_NAME})
message("Build type: " ${CMAKE_BUILD_TYPE})

# Enable CMake support for ASM and C languages
enable_language(C ASM)

# Create an executable object type
add_executable(${CMAKE_PROJECT_NAME})

# Add STM32CubeMX generated sources
add_subdirectory(cmake/stm32cubemx)

# Link directories setup
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined library search paths
)

# Add sources to executable
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user sources here
)

# Add include paths
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined include paths
)

# Add project symbols (macros)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined symbols
)

# Remove wrong libob.a library dependency when using cpp files
list(REMOVE_ITEM CMAKE_C_IMPLICIT_LINK_LIBRARIES ob)

# Add linked libraries
target_link_libraries(${CMAKE_PROJECT_NAME}
    stm32cubemx

    # Add user defined libraries
)

# Generate files commonly used by STM32 programming tools.
set(HEX_FILE "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.hex")
set(BIN_FILE "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.bin")

add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O ihex
            $<TARGET_FILE:${CMAKE_PROJECT_NAME}> ${HEX_FILE}
    COMMAND ${CMAKE_OBJCOPY} -O binary
            $<TARGET_FILE:${CMAKE_PROJECT_NAME}> ${BIN_FILE}
    COMMENT "Generating ${CMAKE_PROJECT_NAME}.hex and ${CMAKE_PROJECT_NAME}.bin"
)

# `cmake --build build --target flash` uses STM32CubeProgrammer.
# macOS programs through ST-Link/SWD. Linux (including Jetson Orin) uses the
# STM32 ROM bootloader over UART and assumes the MCU is already in boot mode.
find_program(STM32_PROGRAMMER_CLI
    NAMES STM32_Programmer_CLI
    HINTS
        "/Applications/STMicroelectronics/STM32Cube/STM32CubeProgrammer/STM32CubeProgrammer.app/Contents/Resources/bin"
        "/Applications/STMicroelectronics/STM32Cube/STM32CubeProgrammer/STM32CubeProgrammer.app/Contents/MacOs/bin"
        "/Applications/STMicroelectronics/STM32Cube/STM32CubeProgrammer/STM32CubeProgrammer.app/Contents/MacOS/bin"
        "/opt/st/stm32cubeprogrammer/bin"
        "/usr/local/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin"
    DOC "Path to STM32CubeProgrammer command-line tool"
)

if(STM32_PROGRAMMER_CLI)
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
        add_custom_target(flash
            COMMAND ${STM32_PROGRAMMER_CLI}
                    -c port=SWD
                    -w $<TARGET_FILE:${CMAKE_PROJECT_NAME}>
                    -v
                    -rst
            DEPENDS ${CMAKE_PROJECT_NAME}
            USES_TERMINAL
            COMMENT "Flashing ${CMAKE_PROJECT_NAME} through ST-Link/SWD"
        )
    elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        set(STM32_UART_PORT "/dev/ttyUSB0" CACHE STRING
            "UART device connected to the STM32 ROM bootloader")
        set(STM32_UART_BAUD "115200" CACHE STRING
            "UART baud rate used by STM32CubeProgrammer")

        add_custom_target(flash
            COMMAND ${STM32_PROGRAMMER_CLI}
                    -c port=${STM32_UART_PORT} br=${STM32_UART_BAUD}
                    -w ${BIN_FILE} 0x08000000
                    -v
                    --start 0x08000000
            DEPENDS ${CMAKE_PROJECT_NAME}
            USES_TERMINAL
            COMMENT "Flashing ${CMAKE_PROJECT_NAME}.bin over ${STM32_UART_PORT}"
        )
    else()
        add_custom_target(flash
            COMMAND ${CMAKE_COMMAND} -E echo
                    "Flashing is configured only for macOS (SWD) and Linux (UART)."
            COMMAND ${CMAKE_COMMAND} -E false
            USES_TERMINAL
            COMMENT "Unsupported host platform"
        )
    endif()
else()
    add_custom_target(flash
        COMMAND ${CMAKE_COMMAND} -E echo
                "STM32_Programmer_CLI was not found. Install STM32CubeProgrammer, then configure again."
        COMMAND ${CMAKE_COMMAND} -E false
        USES_TERMINAL
        COMMENT "STM32CubeProgrammer is required for flashing"
    )
endif()
