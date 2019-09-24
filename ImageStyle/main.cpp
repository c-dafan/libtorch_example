
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <string>
#include <vector>


using namespace std;

int main() {
    string fin = "/home/lab-xjh/app/vgg_feature.pt";
    string content_img_path = "/home/lab-xjh/Pictures/bg.jpeg";
    string style_img_path = "/home/lab-xjh/Pictures/fg.jpeg";
    auto content_img1 = cv::imread(content_img_path);
    cv::resize(content_img1, content_img1, cv::Size(200, 200));
    auto style_img1 = cv::imread(style_img_path);
//    cv::resize(style_img1, style_img1, cv::Size(200, 200));
//    cv::imshow("style", style_img1);
//    cv::imshow("image", content_img1);
//    cv::waitKey();
//    cv::destroyAllWindows();
    torch::Tensor content_img = torch::from_blob(content_img1.data, {1,content_img1.rows, content_img1.cols,3}, torch::kByte);
    content_img = content_img.permute({0,3,1,2});
    content_img = content_img.toType(torch::kFloat);

    auto style_img = torch::from_blob(style_img1.data, {1,content_img1.rows, content_img1.cols,3}, torch::kByte);
    style_img = style_img.permute({0,3,1,2});
    style_img = style_img.toType(torch::kFloat);

    auto opt_img = content_img.clone();
    opt_img.set_requires_grad(true).detach();

    auto model = torch::jit::load(fin);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(content_img);
    auto content_targets = model.forward(inputs).toTuple()->elements();
    inputs.clear();
    inputs.push_back(style_img);
    auto style_targets = model.forward(inputs).toTuple()->elements();
    vector<torch::Tensor> targets{content_targets[0].toTensor(),
                                  content_targets[1].toTensor(),
                                  content_targets[2].toTensor(),
                                  content_targets[3].toTensor(),
                                  content_targets[4].toTensor(),
                                  style_targets[5].toTensor()
    };
    vector<float> weights = {64, 128, 256, 512, 512, 1};
    for (int ii = 0; ii < weights.size(); ii++) {
        float w = weights[ii];
        weights[ii] = (float) 1e3 / (w * w);
    }
    weights[5] = 1;

    auto parm = torch::optim::LBFGSOptions(1);
    auto optimizer = torch::optim::LBFGS(vector<torch::Tensor>{opt_img}, torch::optim::LBFGSOptions(1));
    int n_iter = 0;
    int max_iter = 50;
    function<torch::Tensor()> closure = [&]() {
        optimizer.zero_grad();
        inputs.clear();
        inputs.push_back(opt_img);
        auto out = model.forward(inputs).toTuple()->elements();
        torch::Tensor loss = torch::zeros({});
        for (int ii = 0; ii < out.size(); ii++) {
            loss += torch::mse_loss(out[ii].toTensor()[0], targets[ii][0]) * weights[ii];
        }
        loss.backward();
        n_iter++;
        auto img_tensor = opt_img[0].permute({1, 2, 0});
        img_tensor = (img_tensor - torch::min(img_tensor)) / (torch::max(img_tensor) - torch::min(img_tensor)) * 255;
        img_tensor = img_tensor.toType(torch::kUInt8);
//        img_tensor = img_tensor.toType(torch::kUInt8);
//        cout << img_tensor[0] << endl;
        cv::Mat img_show(200, 200,CV_8UC3,img_tensor.data_ptr());
        cv::imshow("style_image", img_show);
        cv::waitKey(2);
        cout << "iteration: " << n_iter << " ,loss: " << loss << endl;
        return loss;
    };
    auto optimizer2 = torch::optim::LBFGS(vector<torch::Tensor>{opt_img}, torch::optim::LBFGSOptions(0.1));
//    for(int ii=0; ii<max_iter; ii++){
//        optimizer.step(closure);
//    }
    for(int ii=0; ii<max_iter; ii++){
        optimizer2.step(closure);
    }

    return 0;
}
