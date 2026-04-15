cmake_minimum_required(VERSION 3.14)

add_executable(evaluate_ycb_dataset_multiview evaluate_ycb_dataset_multiview.cpp evaluate_ycb_evaluator_multiview.cpp evaluate_ycb_evaluator_multiview.h cosypose.hpp)
target_link_libraries(evaluate_ycb_dataset_multiview PUBLIC m3t PRIVATE mmdeploy "${TORCH_LIBRARIES}")
# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET evaluate_ycb_dataset_multiview
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:evaluate_ycb_dataset_multiview>)
endif (MSVC)

add_executable(evaluate_tless_dataset_multiview evaluate_tless_dataset_multiview.cpp evaluate_tless_evaluator_multiview.cpp evaluate_tless_evaluator_multiview.h cosypose.hpp)
target_link_libraries(evaluate_tless_dataset_multiview PUBLIC m3t PRIVATE mmdeploy "${TORCH_LIBRARIES}")
# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET evaluate_tless_dataset_multiview
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:evaluate_tless_dataset_multiview>)
endif (MSVC)

add_executable(evaluate_scenertm6d_dataset_multiview evaluate_scenertm6d_dataset_multiview.cpp evaluate_scenertm6d_evaluator_multiview.cpp evaluate_scenertm6d_evaluator_multiview.h cosypose.hpp)
target_link_libraries(evaluate_scenertm6d_dataset_multiview PUBLIC m3t PRIVATE mmdeploy "${TORCH_LIBRARIES}")


if (USE_HOLOLENS)
    add_executable(test_hololens test_hololens.cpp)
    target_link_libraries(test_hololens PUBLIC m3t avcodec avformat avutil swresample swscale swscale avfilter)

    add_executable(demo demo.cpp cosypose.hpp)
    target_link_libraries(demo PUBLIC m3t avcodec avformat avutil swresample swscale swscale avfilter PRIVATE mmdeploy ${Pangolin_LIBRARIES} "${TORCH_LIBRARIES}")
    if (MSVC)
      file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
      add_custom_command(TARGET evaluate_ycb_dataset_multiview
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:evaluate_ycb_dataset_multiview>)
    endif (MSVC)
endif ()
