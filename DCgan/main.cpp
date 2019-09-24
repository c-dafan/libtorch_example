#include <iostream>
#include <torch/torch.h>
#include <torch/all.h>
#include <opencv2/opencv.hpp>
#include <time.h>

struct Generator:torch::nn::Module{
    Generator(int d, int nz):
            conv1(torch::nn::Conv2dOptions(nz, d*8, 4).stride(1).padding(0).transposed(true)),
            conv2(torch::nn::Conv2dOptions(d*8, d*4, 4).stride(2).padding(1).transposed(true)),
            conv3(torch::nn::Conv2dOptions(d*4, d*2, 4).stride(2).padding(1).transposed(true)),
            conv4(torch::nn::Conv2dOptions(d*2, d, 4).stride(2).padding(1).transposed(true)),
            conv5(torch::nn::Conv2dOptions(d, 1, 4).stride(2).padding(1).transposed(true)),
            conv1_bn(torch::nn::BatchNormOptions(d*8)),
            conv2_bn(torch::nn::BatchNormOptions(d*4)),
            conv3_bn(torch::nn::BatchNormOptions(d*2)),
            conv4_bn(torch::nn::BatchNormOptions(d))
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("conv5", conv5);
        register_module("conv4_bn", conv4_bn);
        register_module("conv3_bn", conv3_bn);
        register_module("conv2_bn", conv2_bn);
        register_module("conv1_bn", conv1_bn);

    }
    torch::Tensor forward(torch::Tensor x){
        x = torch::relu(conv1_bn(conv1(x)));
        x = torch::relu(conv2_bn(conv2(x)));
        x = torch::relu(conv3_bn(conv3(x)));
        x = torch::relu(conv4_bn(conv4(x)));
        x = torch::tanh(conv5(x));
        return x;
    }
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm conv1_bn;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm conv2_bn;
    torch::nn::Conv2d conv3;
    torch::nn::BatchNorm conv3_bn;
    torch::nn::Conv2d conv4;
    torch::nn::BatchNorm conv4_bn;
    torch::nn::Conv2d conv5;
};

struct Discriminator:torch::nn::Module{
    Discriminator(int d):
            conv1(torch::nn::Conv2dOptions(1, d, 4).padding(1).stride(2)),
            conv2(torch::nn::Conv2dOptions(d, d*2, 4).padding(1).stride(2)),
            conv2_bn(torch::nn::BatchNormOptions(d*2)),
            conv3(torch::nn::Conv2dOptions(d*2, d*4, 4).padding(1).stride(2)),
            conv3_bn(torch::nn::BatchNormOptions(d*4)),
            conv4(torch::nn::Conv2dOptions(d*4, d*8, 4).padding(1).stride(2)),
            conv4_bn(torch::nn::BatchNormOptions(d*8)),
            conv5(torch::nn::Conv2dOptions(d*8, 1, 4).stride(2)) {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv5", conv5);
        register_module("conv4", conv4);
        register_module("conv3", conv3);
        register_module("conv2_bn", conv2_bn);
        register_module("conv3_bn", conv3_bn);
        register_module("conv4_bn", conv4_bn);

    }
    torch::Tensor forward(torch::Tensor x){
        x = torch::leaky_relu(conv1(x), 0.2);
        x = torch::leaky_relu(conv2_bn(conv2(x)), 0.2);
        x = torch::leaky_relu(conv3_bn(conv3(x)), 0.2);
        x = torch::leaky_relu(conv4_bn(conv4(x)), 0.2);
        x = conv5(x);
        x = torch::sigmoid(x);
        return x;
    }
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm conv2_bn;
    torch::nn::Conv2d conv2;
    torch::nn::Conv2d conv3;
    torch::nn::BatchNorm conv3_bn;
    torch::nn::Conv2d conv4;
    torch::nn::BatchNorm conv4_bn;
    torch::nn::Conv2d conv5;
};

int main() {
    torch::DeviceType device_type;
    std::string root = "/home/lab-xjh/app/data/";
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    Generator GNet(128, 100);
    GNet.to(device);

    Discriminator DNet(128);
    DNet.to(device);

    torch::optim::Adam optimizerD(DNet.parameters(), torch::optim::AdamOptions(0.0002).beta1(0.5));
    torch::optim::Adam optimizerG(GNet.parameters(), torch::optim::AdamOptions(0.0002).beta1(0.5));

    std::function<torch::data::Example<>(torch::data::Example<>)>
            resize = [](torch::data::Example<> inputs){
        auto data = inputs.data;
        data *= 255;
        data = data.toType(torch::kUInt8);
        data = data.permute({1, 2, 0});
        cv::Mat out;
        cv::resize(cv::Mat(28, 28, CV_8UC1, data.data_ptr()), out, cv::Size_<int>(64, 64));
        data = torch::from_blob(out.data, {64, 64, 1}, torch::kByte);
        data = data.permute({2, 0, 1});
        data = data.toType(torch::kFloat);
        data /= 255;
        return torch::data::Example<>(data, inputs.target);
    };

    auto train_data = torch::data::datasets::MNIST(root, torch::data::datasets::MNIST::Mode::kTrain)
             .map(torch::data::transforms::Lambda<torch::data::Example<>>(resize))
                        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                        .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::DistributedRandomSampler >(std::move(train_data),
                        torch::data::DataLoaderOptions(64));

    for(int epoch=0; epoch<20; epoch++){
        for(auto& batch: *train_loader){
            std::clock_t start = std::clock();
            DNet.zero_grad();
            auto x = batch.data.to(device);
            auto batch_size = x.size(0);
            auto y_real_ = torch::ones({batch_size}).to(device);
            auto y_fake_ = torch::zeros({batch_size}).to(device);
            auto out = DNet.forward(x);
            auto lossD_real = torch::binary_cross_entropy(out, y_real_);

            auto noise = torch::randn({batch_size,100,1,1}).to(device);
            auto x_fake = GNet.forward(noise);
            auto out_fake = DNet.forward(x_fake);
            auto lossD_fake = torch::binary_cross_entropy(out_fake, y_fake_);
            auto lossD = (lossD_fake + lossD_real);
            lossD.backward();
            optimizerD.step();

            GNet.zero_grad();
            auto noise2 = torch::randn({batch_size, 100,1,1}).to(device);
            auto z_fake = GNet.forward(noise2);
            auto out_real = DNet.forward(z_fake);
            auto errG = torch::binary_cross_entropy(out_real, y_real_);
            errG.backward();
            optimizerG.step();
            std::cout << epoch << "/ 20 , loss_d: " << lossD << "  loss_g: "<<errG << std::endl;
            std::cout <<  (std::clock() - start)/1000 << std::endl;
//            auto noise3 = torch::randn({1, 100,1,1}).to(device);
//            auto fake_img = GNet.forward(noise2)[0];
//            fake_img = fake_img.permute({1, 2, 0});
//            fake_img *= 255;
//            fake_img = fake_img.toType(torch::kUInt8);

//            cv::imshow("img", cv::Mat(64, 64, CV_8UC1, fake_img.data_ptr()));
//            cv::waitKey(2);
        }
    }

    return 0;
}