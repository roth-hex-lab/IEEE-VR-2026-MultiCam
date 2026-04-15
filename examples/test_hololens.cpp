#include <filesystem/filesystem.h>
#include <m3t/hololens_camera.h>
#include <m3t/image_viewer.h>
#include <m3t/tracker.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <memory>
#include <string>



hl2ss::umq_command_buffer buffer;
char const *host = "192.168.0.149";
std::unique_ptr<hl2ss::ipc_umq> client_umq =
    hl2ss::lnm::ipc_umq(host, hl2ss::ipc_port::UNITY_MESSAGE_QUEUE);

bool is_open = false;


void test_server() {
  
  if (is_open == false) {
     client_umq->open();
    is_open == true;
  }
  buffer.clear();
  std::string test = "test";
  std::vector<uint8_t> data(test.begin(), test.end());
  buffer.add(0xFFFFFFFE, data.data(), data.size());
  
  try {
    client_umq->push(buffer.get_data(), buffer.get_size());
    std::vector<uint32_t> response;
    response.resize(buffer.get_count());
    client_umq->pull(response.data(), buffer.get_count());
    std::cout << "Response from HoloLens app: " << response[0] << std::endl;
  } catch (...) {
    std::cerr << "server connection failed" << std::endl;
  }
}


int main(int argc, char *argv[]) {
  //hl2ss::client::initialize();
  //const std::filesystem::path sequence_directory{argv[1]};

  
  auto hololens_color_ptr{
      std::make_shared<m3t::HololensColorCamera>(host, "hololens_color")};
  /*
  auto hololens_depth_ptr{
      std::make_shared<m3t::HololensDepthCamera>(host, "hololens_depth")};*/
  //hololens_color_ptr->StartSavingImages(sequence_directory);
  //hololens_depth_ptr->StartSavingImages(sequence_directory);

  auto color_viewer_ptr{std::make_shared<m3t::ImageColorViewer>(
      "color_viewer", hololens_color_ptr)};
  /*
  auto depth_viewer_ptr{std::make_shared<m3t::ImageDepthViewer>(
      "depth_viewer", hololens_depth_ptr, 0.1f, 2.0f)};*/
  //client = hl2ss::lnm::ipc_umq(host, hl2ss::ipc_port::UNITY_MESSAGE_QUEUE);
  
  auto tracker_ptr{std::make_shared<m3t::Tracker>("tracker")};
  tracker_ptr->AddViewer(color_viewer_ptr);
  //tracker_ptr->AddViewer(depth_viewer_ptr);
  
  
  //std::cout << "good";

  if (!tracker_ptr->SetUp()) return -1;
  //std::thread server(test_server);
  //server.detach();
  //_sleep(1000);
  
  if (!tracker_ptr->RunTrackerProcess(false, false)) return -1;


 
  return 0;
}