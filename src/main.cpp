#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include <cmath>
#include <cassert>
#include <string>
#include <iomanip>
#include <algorithm>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"


#define DEBUG_FLAG              1       // Debug flag for image channels
#define MIN_SOMA_SIZE           20      // Min soma size
#define CELL_COVERAGE_RATIO     0.5     // Coverage ratio
#define MIN_ARC_LENGTH_FILTER   10      // Min arc length filter threshold
#define SOMA_COVERAGE_RATIO     0.5     // Coverage ratio
#define SOMA_FACTOR             1.6     // Soma radius = factor * nuclues radius
#define PI                      3.14    // Approximate value of pi
#define NUM_AREA_BINS           21      // Number of bins
#define BIN_AREA                25      // Bin area
#define NUM_Z_LAYERS_COMBINED   3       // Number of z-layers combined


/* Channel type */
enum class ChannelType : unsigned char {
    BLUE = 0,
    PURPLE,
    RED
};

/* Hierarchy type */
enum class HierarchyType : unsigned char {
    INVALID_CNTR = 0,
    CHILD_CNTR,
    PARENT_CNTR
};

/* Enhance the image */
// Enhance the image using Gaussian blur and thresholding
bool enhanceImage(cv::Mat src, ChannelType channel_type, cv::Mat *dst) {

    cv::Mat enhanced;
    switch(channel_type) {
        case ChannelType::BLUE: {
            cv::threshold(src, enhanced, 150, 255, cv::THRESH_BINARY);
        } break;

        case ChannelType::PURPLE: {
            cv::Mat temp, gray;
            cv::threshold(src, temp, 50, 255, cv::THRESH_TOZERO);
            cv::threshold(temp, temp, 150, 255, cv::THRESH_TOZERO_INV);
            cv::threshold(temp, enhanced, 50, 255, cv::THRESH_BINARY);
        } break;

        case ChannelType::RED: {
            cv::threshold(src, enhanced, 170, 255, cv::THRESH_BINARY);
        } break;

        default: {
            std::cerr << "Invalid channel type" << std::endl;
            return false;
        }
    }
    *dst = enhanced;
    return true;
}

/* Find the contours in the image */
void contourCalc(   cv::Mat src,
                    double min_area, cv::Mat *dst, 
                    std::vector<std::vector<cv::Point>> *contours, 
                    std::vector<cv::Vec4i> *hierarchy, 
                    std::vector<HierarchyType> *validity_mask, 
                    std::vector<double> *parent_area    ) {

    cv::Mat temp_src;
    src.copyTo(temp_src);
    findContours(temp_src, *contours, *hierarchy, cv::RETR_EXTERNAL, 
                                                        cv::CHAIN_APPROX_SIMPLE);
    *dst = cv::Mat::zeros(temp_src.size(), CV_8UC3);
    if (!contours->size()) return;
    validity_mask->assign(contours->size(), HierarchyType::INVALID_CNTR);
    parent_area->assign(contours->size(), 0.0);

    // Keep the contours whose size is >= than min_area
    cv::RNG rng(12345);
    for (int index = 0 ; index < (int)contours->size(); index++) {
        if ((*hierarchy)[index][3] > -1) continue; // ignore child
        auto cntr_external = (*contours)[index];
        double area_external = fabs(contourArea(cv::Mat(cntr_external)));
        if (area_external < min_area) continue;

        std::vector<int> cntr_list;
        cntr_list.push_back(index);

        int index_hole = (*hierarchy)[index][2];
        double area_hole = 0.0;
        while (index_hole > -1) {
            std::vector<cv::Point> cntr_hole = (*contours)[index_hole];
            double temp_area_hole = fabs(contourArea(cv::Mat(cntr_hole)));
            if (temp_area_hole) {
                cntr_list.push_back(index_hole);
                area_hole += temp_area_hole;
            }
            index_hole = (*hierarchy)[index_hole][0];
        }
        double area_contour = area_external - area_hole;
        if (area_contour >= min_area) {
            (*validity_mask)[cntr_list[0]] = HierarchyType::PARENT_CNTR;
            (*parent_area)[cntr_list[0]] = area_contour;
            for (unsigned int i = 1; i < cntr_list.size(); i++) {
                (*validity_mask)[cntr_list[i]] = HierarchyType::CHILD_CNTR;
            }
            cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0,255), 
                                            rng.uniform(0,255));
            drawContours(*dst, *contours, index, color, CV_FILLED, 8, *hierarchy);
        }
    }
}

/* Filter out ill-formed or small cells */
void filterCells(   ChannelType channel_type,
                    cv::Mat channel,
                    std::vector<std::vector<cv::Point>> contours,
                    std::vector<HierarchyType> contour_mask,
                    std::vector<std::vector<cv::Point>> *filtered_contours ) {

    for (size_t i = 0; i < contours.size(); i++) {
        if (contour_mask[i] != HierarchyType::PARENT_CNTR) continue;
        // Eliminate extremely small contours
        auto arc_length = arcLength(contours[i], true);
        if ((contours[i].size() < 5) || (arc_length < MIN_ARC_LENGTH_FILTER)) continue;

        switch(channel_type) {
            case ChannelType::BLUE: {
                // Calculate center of the nucleus
                cv::RotatedRect min_area_rect = minAreaRect(cv::Mat(contours[i]));
                float aspect_ratio = float(min_area_rect.size.width)/min_area_rect.size.height;
                if (aspect_ratio > 1.0) {
                    aspect_ratio = 1.0/aspect_ratio;
                }
                if (aspect_ratio >= CELL_COVERAGE_RATIO) {
                    filtered_contours->push_back(contours[i]);
                }
            } break;

            case ChannelType::PURPLE: 
            case ChannelType::RED: {
                filtered_contours->push_back(contours[i]);
            } break;
        }
    }
}

/* Find cell soma */
bool findCellSoma( std::vector<cv::Point> nucleus_contour, 
                   cv::Mat cell_mask, 
                   cv::Mat *intersection, 
                   std::vector<cv::Point> *soma_contour ) {

    bool status = false;

    // Calculate radius and center of the nucleus
    cv::Moments mu = moments(nucleus_contour, true);
    cv::Point2f mc = cv::Point2f(   static_cast<float>(mu.m10/mu.m00), 
                                    static_cast<float>(mu.m01/mu.m00)   );
    cv::RotatedRect min_area_rect = minAreaRect(cv::Mat(nucleus_contour));
    float nucleus_radius = sqrt(min_area_rect.size.width * min_area_rect.size.height);

    // Nucleus' region of influence
    cv::Mat roi_mask = cv::Mat::zeros(cell_mask.size(), CV_8UC1);
    float roi_radius = (float) (SOMA_FACTOR * nucleus_radius);
    cv::circle(roi_mask, mc, roi_radius, 255, -1, 8);
    cv::circle(roi_mask, mc, nucleus_radius, 0, -1, 8);
    int circle_score = countNonZero(roi_mask);

    // Soma present in ROI
    bitwise_and(roi_mask, cell_mask, *intersection);
    int intersection_score = countNonZero(*intersection);

    // Add the nuecleus contour to intersection region
    cv::circle(*intersection, mc, nucleus_radius, 255, -1, 8);

    // Add to the soma mask if coverage area exceeds a certain threshold
    float ratio = ((float) intersection_score) / circle_score;
    if (ratio >= SOMA_COVERAGE_RATIO) {

        // Segment
        cv::Mat soma_segmented;
        std::vector<std::vector<cv::Point>> contours_soma;
        std::vector<cv::Vec4i> hierarchy_soma;
        std::vector<HierarchyType> soma_contour_mask;
        std::vector<double> soma_contour_area;
        contourCalc(    *intersection, 
                        1.0, 
                        &soma_segmented, 
                        &contours_soma, 
                        &hierarchy_soma, 
                        &soma_contour_mask, 
                        &soma_contour_area
                   );

        double max_area  = 0.0;
        for (size_t i = 0; i < contours_soma.size(); i++) {
            if (soma_contour_mask[i] != HierarchyType::PARENT_CNTR) continue;
            if (contours_soma[i].size() < 5) continue;
            if (soma_contour_area[i] < MIN_SOMA_SIZE) continue;

            // Find the largest permissible contour
            if (soma_contour_area[i] > max_area) {
                max_area = soma_contour_area[i];
                *soma_contour = contours_soma[i];
                status = true;
            }
        }
    }
    return status;
}

/* Separation metrics */
void separationMetrics( std::vector<std::vector<cv::Point>> contours, 
                        float *mean_diameter,
                        float *stddev_diameter,
                        float *mean_aspect_ratio,
                        float *stddev_aspect_ratio,
                        float *mean_error_ratio,
                        float *stddev_error_ratio ) {

    // Compute the normal distribution parameters of cells
    std::vector<cv::Point2f> mc(contours.size());
    std::vector<float> dia(contours.size());
    std::vector<float> aspect_ratio(contours.size());
    std::vector<float> error_ratio(contours.size());

    for (size_t i = 0; i < contours.size(); i++) {
        cv::Moments mu = moments(contours[i], true);
        mc[i] = cv::Point2f(static_cast<float>(mu.m10/mu.m00), 
                                            static_cast<float>(mu.m01/mu.m00));
        cv::RotatedRect min_area_rect = minAreaRect(cv::Mat(contours[i]));
        aspect_ratio[i] = float(min_area_rect.size.width)/min_area_rect.size.height;
        if (aspect_ratio[i] > 1.0) {
            aspect_ratio[i] = 1.0/aspect_ratio[i];
        }
        float actual_area = contourArea(contours[i]);
        dia[i] = 2 * sqrt(actual_area / PI);
        float ellipse_area = 
            (float) (PI * min_area_rect.size.width * min_area_rect.size.height);
        error_ratio[i] = (ellipse_area - actual_area) / ellipse_area;
    }
    cv::Scalar mean_dia, stddev_dia;
    cv::meanStdDev(dia, mean_dia, stddev_dia);
    *mean_diameter = static_cast<float>(mean_dia.val[0]);
    *stddev_diameter = static_cast<float>(stddev_dia.val[0]);

    cv::Scalar mean_ratio, stddev_ratio;
    cv::meanStdDev(aspect_ratio, mean_ratio, stddev_ratio);
    *mean_aspect_ratio = static_cast<float>(mean_ratio.val[0]);
    *stddev_aspect_ratio = static_cast<float>(stddev_ratio.val[0]);

    cv::Scalar mean_err_ratio, stddev_err_ratio;
    cv::meanStdDev(error_ratio, mean_err_ratio, stddev_err_ratio);
    *mean_error_ratio = static_cast<float>(mean_err_ratio.val[0]);
    *stddev_error_ratio = static_cast<float>(stddev_err_ratio.val[0]);
}

/* Group contour areas into bins */
void binArea(   std::vector<HierarchyType> contour_mask, 
                std::vector<double> contour_area, 
                std::string *contour_output ) {

    std::vector<unsigned int> count(NUM_AREA_BINS, 0);
    for (size_t i = 0; i < contour_mask.size(); i++) {
        if (contour_mask[i] != HierarchyType::PARENT_CNTR) continue;
        unsigned int area = static_cast<unsigned int>(round(contour_area[i]));
        unsigned int bin_index = 
            (area/BIN_AREA < NUM_AREA_BINS) ? area/BIN_AREA : NUM_AREA_BINS-1;
        count[bin_index]++;
    }

    unsigned int contour_cnt = 0;
    std::string area_binned;
    for (size_t i = 0; i < count.size(); i++) {
        area_binned += "," + std::to_string(count[i]);
        contour_cnt += count[i];
    }
    *contour_output = std::to_string(contour_cnt) + area_binned;
}

/* Process the images inside each directory */
bool processDir(std::string path, std::string image_name, std::string metrics_file) {

    /* Create the data output file for images that were processed */
    std::ofstream data_stream;
    data_stream.open(metrics_file, std::ios::app);
    if (!data_stream.is_open()) {
        std::cerr << "Could not open the data output file." << std::endl;
        return false;
    }

    // Create the output directory
    std::string out_directory = path + "result/";
    struct stat st = {0};
    if (stat(out_directory.c_str(), &st) == -1) {
        mkdir(out_directory.c_str(), 0700);
    }
    out_directory = out_directory + image_name + "/";
    st = {0};
    if (stat(out_directory.c_str(), &st) == -1) {
        mkdir(out_directory.c_str(), 0700);
    }

    // Count the number of images
    std::string dir_name = path + "jpg/" + image_name + "/";
    DIR *read_dir = opendir(dir_name.c_str());
    if (!read_dir) {
        std::cerr << "Could not open directory '" << dir_name << "'" << std::endl;
        return false;
    }
    struct dirent *dir = NULL;
    uint8_t z_count = 0;
    bool collect_name_pattern = false;
    std::string end_pattern;
    while ((dir = readdir(read_dir))) {
        if (!strcmp (dir->d_name, ".") || !strcmp (dir->d_name, "..")) continue;
        if (!collect_name_pattern) {
            std::string delimiter = "c1+";
            end_pattern = dir->d_name;
            size_t pos = end_pattern.find(delimiter);
            end_pattern.erase(0, pos);
            collect_name_pattern = true;
        }
        z_count++;
    }

    std::vector<cv::Mat>    blue_list(NUM_Z_LAYERS_COMBINED), 
                            green_list(NUM_Z_LAYERS_COMBINED), 
                            red_list(NUM_Z_LAYERS_COMBINED);
    for (uint8_t z_index = 1; z_index <= z_count; z_index++) {

        // Create the input filename and rgb stream output filenames
        std::string in_filename;
        if (z_count < 10) {
            in_filename = dir_name + image_name + 
                                        "_z" + std::to_string(z_index) + end_pattern;
        } else {
            if (z_index < 10) {
                in_filename = dir_name + image_name + 
                                        "_z0" + std::to_string(z_index) + end_pattern;
            } else if (z_index < 100) {
                in_filename = dir_name + image_name + 
                                        "_z" + std::to_string(z_index) + end_pattern;
            } else { // assuming number of z plane layers will never exceed 99
                std::cerr << "Does not support more than 99 z layers curently" << std::endl;
                return false;
            }
        }

        // Extract the bgr streams for each input image
        cv::Mat img = cv::imread(in_filename.c_str(), -1);
        if (img.empty()) return false;

        // Original image
        std::string out_original = out_directory + "zlayer_" + 
                                        std::to_string(z_index) + "_a_original.jpg";
        if (!DEBUG_FLAG) cv::imwrite(out_original.c_str(), img);

        std::vector<cv::Mat> channel(3);
        cv::split(img, channel);
        blue_list[(z_index-1)%NUM_Z_LAYERS_COMBINED]  = channel[0];
        green_list[(z_index-1)%NUM_Z_LAYERS_COMBINED] = channel[1];
        red_list[(z_index-1)%NUM_Z_LAYERS_COMBINED]   = channel[2];

        // Continue collecting layers if needed
        //if (z_index%NUM_Z_LAYERS_COMBINED && (z_index != z_count)) continue;
        if (z_index < NUM_Z_LAYERS_COMBINED) continue;

        data_stream << image_name << ","
                    << std::to_string(z_index - NUM_Z_LAYERS_COMBINED + 1) << ","
                    << std::to_string(z_index) << ",";

        // Merge some layers together
        cv::Mat blue  = cv::Mat::zeros(channel[0].size(), CV_8UC1);
        cv::Mat green = cv::Mat::zeros(channel[1].size(), CV_8UC1);
        cv::Mat red   = cv::Mat::zeros(channel[1].size(), CV_8UC1);
        for (unsigned int merge_index = 0; 
                    merge_index < NUM_Z_LAYERS_COMBINED; merge_index++) {
            bitwise_or(blue, blue_list[merge_index], blue);
            bitwise_or(green, green_list[merge_index], green);
            bitwise_or(red, red_list[merge_index], red);
        }

        /** Gather BGR channel information needed for feature extraction **/

        /* Blue channel */
        cv::Mat blue_enhanced;
        if(!enhanceImage(blue, ChannelType::BLUE, &blue_enhanced)) return false;
        std::string out_blue = out_directory + "zlayer_" + 
                                        std::to_string(z_index) + "_blue_enhanced.jpg";
        if (DEBUG_FLAG) cv::imwrite(out_blue.c_str(), blue_enhanced);

        // Segment the blue channel
        cv::Mat blue_segmented;
        std::vector<std::vector<cv::Point>> contours_blue;
        std::vector<cv::Vec4i> hierarchy_blue;
        std::vector<HierarchyType> blue_contour_mask;
        std::vector<double> blue_contour_area;
        contourCalc(    blue_enhanced,
                        1.0,
                        &blue_segmented, 
                        &contours_blue,
                        &hierarchy_blue, 
                        &blue_contour_mask,
                        &blue_contour_area  );
        std::vector<std::vector<cv::Point>> contours_blue_filtered;
        filterCells(    ChannelType::BLUE, 
                        blue_enhanced, 
                        contours_blue, 
                        blue_contour_mask, 
                        &contours_blue_filtered );

        /* Purple channel */
        cv::Mat purple_enhanced;
        if(!enhanceImage(red, ChannelType::PURPLE, &purple_enhanced)) return false;
        std::string out_purple = out_directory + "zlayer_" + 
                                        std::to_string(z_index) + "_purple_enhanced.jpg";
        if (DEBUG_FLAG) cv::imwrite(out_purple.c_str(), purple_enhanced);

        /* Red channel */
        cv::Mat red_enhanced;
        if(!enhanceImage(red, ChannelType::RED, &red_enhanced)) return false;
        std::string out_red = out_directory + "zlayer_" + 
                                        std::to_string(z_index) + "_red_enhanced.jpg";
        if (DEBUG_FLAG) cv::imwrite(out_red.c_str(), red_enhanced);

        // Segment the red channel
        cv::Mat red_segmented;
        std::vector<std::vector<cv::Point>> contours_red;
        std::vector<cv::Vec4i> hierarchy_red;
        std::vector<HierarchyType> red_contour_mask;
        std::vector<double> red_contour_area;
        contourCalc(    red_enhanced,
                        1.0,
                        &red_segmented, 
                        &contours_red,
                        &hierarchy_red, 
                        &red_contour_mask,
                        &red_contour_area  );
        std::vector<std::vector<cv::Point>> contours_red_filtered;
        filterCells(    ChannelType::RED, 
                        red_enhanced, 
                        contours_red, 
                        red_contour_mask, 
                        &contours_red_filtered  );


        /** Classify the cell soma **/

        std::vector<std::vector<cv::Point>> contours_purple;
        cv::Mat purple_intersection = cv::Mat::zeros(purple_enhanced.size(), CV_8UC1);
        for (size_t i = 0; i < contours_blue_filtered.size(); i++) {
            std::vector<cv::Point> purple_contour;
            cv::Mat temp;
            if (findCellSoma( contours_blue_filtered[i], purple_enhanced, &temp, &purple_contour )) {
                contours_purple.push_back(purple_contour);
                bitwise_or(purple_intersection, temp, purple_intersection);
                cv::Mat temp_not;
                bitwise_not(temp, temp_not);
                bitwise_and(purple_enhanced, temp_not, purple_enhanced);
            }
        }


        /** Collect the metrics **/

        float mean_dia = 0.0, stddev_dia = 0.0;
        float mean_aspect_ratio = 0.0, stddev_aspect_ratio = 0.0;
        float mean_error_ratio = 0.0, stddev_error_ratio = 0.0;
        separationMetrics(  contours_purple, 
                            &mean_dia, 
                            &stddev_dia, 
                            &mean_aspect_ratio, 
                            &stddev_aspect_ratio, 
                            &mean_error_ratio, 
                            &stddev_error_ratio
                        );
        data_stream << contours_purple.size() << "," 
                    << mean_dia << "," 
                    << stddev_dia << "," 
                    << mean_aspect_ratio << "," 
                    << stddev_aspect_ratio << "," 
                    << mean_error_ratio << "," 
                    << stddev_error_ratio << ",";

        data_stream << std::endl;
    }
    closedir(read_dir);
    data_stream.close();

    return true;
}

/* Main - create the threads and start the processing */
int main(int argc, char *argv[]) {

    /* Check for argument count */
    if (argc != 2) {
        std::cerr << "Invalid number of arguments." << std::endl;
        return -1;
    }

    /* Read the path to the data */
    std::string path(argv[1]);

    /* Read the list of directories to process */
    std::string image_list_filename = path + "image_list.dat";
    std::vector<std::string> input_images;
    FILE *file = fopen(image_list_filename.c_str(), "r");
    if (!file) {
        std::cerr << "Could not open 'image_list.dat' inside '" << path << "'." << std::endl;
        return -1;
    }
    char line[128];
    while (fgets(line, sizeof(line), file) != NULL) {
        line[strlen(line)-1] = 0;
        std::string temp_str(line);
        input_images.push_back(temp_str);
    }
    fclose(file);

    /* Create and prepare the file for metrics */
    std::string metrics_file = path + "computed_metrics.csv";
    std::ofstream data_stream;
    data_stream.open(metrics_file, std::ios::out);
    if (!data_stream.is_open()) {
        std::cerr << "Could not create the metrics file." << std::endl;
        return -1;
    }
    data_stream << "CZI Image,";
    data_stream << "Z-index start,";
    data_stream << "Z-index end,";
    data_stream << "Neural Cell Count,";
    data_stream << "Neural Soma Diameter (mean),";
    data_stream << "Neural Soma Diameter (std. dev.),";
    data_stream << "Neural Soma Aspect Ratio (mean),";
    data_stream << "Neural Soma Aspect Ratio (std. dev.),";
    data_stream << "Neural Soma Error Ratio (mean),";
    data_stream << "Neural Soma Error Ratio (std. dev.),";
    data_stream << std::endl;
    data_stream.close();

    /* Process each image */
    for (unsigned int index = 0; index < input_images.size(); index++) {
        std::cout << "Processing " << input_images[index] << std::endl;
        if (!processDir(path, input_images[index], metrics_file)) {
            std::cout << "ERROR !!!" << std::endl;
            return -1;
        }
    }

    return 0;
}

