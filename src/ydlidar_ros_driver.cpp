//
// The MIT License (MIT)
//
// Copyright (c) 2020 EAIBOT. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

// ---- ROS ----
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_srvs/Empty.h>
#include <cube_msgs/VehicleState.h>
#include <geometry_msgs/Polygon.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <geometry_msgs/Pose2D.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <tf/transform_listener.h>
#include <laser_geometry/laser_geometry.h>
#include <pcl_ros/transforms.h>

// ---- PCL ----
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>

// ---- YDLIDAR SDK ----
#include "src/CYdLidar.h"
#include "ydlidar_config.h"

// ---- STL ----
#include <atomic>
#include <cmath>
#include <limits>
#include <mutex>

// =============================================================================
// Constants
// =============================================================================

#define YDLIDAR_ROS_VERSION "1.0.2"

static constexpr int   kMaxRestartAttempts  = 5;
static constexpr float kSORMeanK            = 12.0f;
static constexpr float kSORStddevThresh     = 0.5f;
static constexpr int   kDownsampleStride    = 2;
static constexpr float kTFTimeoutSec        = 0.1f;

struct FilterZone
{
    float center_rad;
    float half_width_rad;
};

// =============================================================================
// Global state
// =============================================================================

CYdLidar  laser;
bool      is_paused   = false;
int       retry_count = 0;
ros::Time lastRestart;

// Filter state updated from ROS topics/timer (accessed from main loop thread)
struct FilterConfig
{
    std::vector<std::pair<float, float>> base_polygon;  // cart polygon, unrotated (meters)
    float lateral_scale = 1.0f;
    std::vector<FilterZone> filter_zones;
};
FilterConfig       g_filter_config;
std::mutex         g_config_mutex;
std::atomic<float>   g_stage_angle{0.0f};
std::atomic<uint8_t> g_drive_mode{cube_msgs::VehicleState::MODE_LOCKED};

geometry_msgs::Pose2D g_door_pose;
bool                  g_has_door_pose = false;
float                 g_door_size = 2.0f;
float                 g_door_filter_max_distance = 2.5f;

// =============================================================================
// Geometry helpers
// =============================================================================

static float normalizeTo2pi(float angle)
{
    const float two_pi = 2.0f * static_cast<float>(M_PI);
    angle = std::fmod(angle, two_pi);
    if (angle < 0.0f) angle += two_pi;
    return angle;
}

// Ray-casting algorithm: returns true if (x, y) is inside the polygon.
static bool isPointInPolygon(float x, float y, const std::vector<std::pair<float, float>>& polygon)
{
    int crossings = 0;
    const int size = static_cast<int>(polygon.size());
    
	for (int i = 0, j = size - 1; i < size; j = i++) {
        const float xi = polygon[i].first;  
		const float yi = polygon[i].second;
        const float xj = polygon[j].first; 
		const float yj = polygon[j].second;
        if ((yi > y) != (yj > y)) {
            const float intersection_x = (xj - xi) * (y - yi) / (yj - yi) + xi;
            if (x < intersection_x) ++crossings;
        }
    }
    return (crossings % 2) == 1;
}

// Returns true if the point falls inside any of the configured filter zones.
// Each zone center is defined relative to stage_angle (in base_link: center_rad + stage_angle).
static bool isPointInConicFilterArea(float x, float y, float stage_angle, const std::vector<FilterZone>& zones)
{
    if (zones.empty()) return false;

    const float ray_angle        = normalizeTo2pi(std::atan2(y, x));
    const float stage_normalized = normalizeTo2pi(stage_angle);

    for (const auto& zone : zones) {
        const float left_boundary  = normalizeTo2pi(zone.center_rad + zone.half_width_rad + stage_normalized);
        const float right_boundary = normalizeTo2pi(zone.center_rad - zone.half_width_rad + stage_normalized);

        const bool wraps = right_boundary > left_boundary;
        const bool in_zone = wraps
            ? (ray_angle > right_boundary) || (ray_angle < left_boundary)
            : (ray_angle > right_boundary) && (ray_angle < left_boundary);

        if (in_zone) return true;
    }
    return false;
}

// =============================================================================
// Visualization helper
// =============================================================================

static visualization_msgs::MarkerArray makePolygonMarkers(
    const std::vector<std::pair<float, float>>& polygon,
    const std_msgs::Header& header)
{
    visualization_msgs::Marker line;
    line.header             = header;
    line.ns                 = "obstacle_polygon";
    line.id                 = 0;
    line.type               = visualization_msgs::Marker::LINE_STRIP;
    line.action             = visualization_msgs::Marker::ADD;
    line.scale.x            = 0.02;
    line.pose.orientation.w = 1.0;
    line.lifetime           = ros::Duration(0.3);
    line.color.r            = 0.1f;
    line.color.g            = 0.8f;
    line.color.b            = 1.0f;
    line.color.a            = 1.0f;

    for (const auto& p : polygon) {
        geometry_msgs::Point gp;
        gp.x = p.first; gp.y = p.second;
        line.points.push_back(gp);
    }
    // Close the polygon loop
    if (!polygon.empty())
        line.points.push_back(line.points.front());

    visualization_msgs::MarkerArray out;
    out.markers.push_back(std::move(line));
    return out;
}


static visualization_msgs::MarkerArray makeDoorPolygonMarkers(
    const std::vector<std::pair<float, float>>& polygon,
    const std_msgs::Header& header)
{
    visualization_msgs::MarkerArray out;
    visualization_msgs::Marker m;
    m.header             = header;
    m.ns                 = "door_polygon";
    m.id                 = 0;
    m.pose.orientation.w = 1.0;

    if (polygon.empty()) {
        m.action = visualization_msgs::Marker::DELETE;
        out.markers.push_back(m);
        return out;
    }

    m.type     = visualization_msgs::Marker::LINE_STRIP;
    m.action   = visualization_msgs::Marker::ADD;
    m.scale.x  = 0.05f;
    m.lifetime = ros::Duration(0.3);
    m.color.r  = 1.0f;
    m.color.g  = 0.5f;
    m.color.b  = 0.0f;
    m.color.a  = 1.0f;

    for (const auto& p : polygon) {
        geometry_msgs::Point gp;
        gp.x = p.first;
        gp.y = p.second;
        m.points.push_back(gp);
    }
    m.points.push_back(m.points.front());

    out.markers.push_back(std::move(m));
    return out;
}


static visualization_msgs::MarkerArray makeFilterZoneMarkers(
    const std::vector<FilterZone>& zones,
    float stage_angle,
    const std_msgs::Header& header)
{
    visualization_msgs::MarkerArray out;
    if (zones.empty()) return out;

    constexpr float RADIUS    = 2.0f;
    constexpr int   ARC_STEPS = 20;

    for (std::size_t i = 0; i < zones.size(); ++i)
    {
        const float center      = stage_angle + zones[i].center_rad;
        const float right_angle = center - zones[i].half_width_rad;
        const float left_angle  = center + zones[i].half_width_rad;

        visualization_msgs::Marker m;
        m.header             = header;
        m.ns                 = "filter_zones";
        m.id                 = static_cast<int>(i);
        m.type               = visualization_msgs::Marker::LINE_STRIP;
        m.action             = visualization_msgs::Marker::ADD;
        m.pose.orientation.w = 1.0;
        m.lifetime           = ros::Duration(0.3);
        m.scale.x            = 0.02f;
        m.color.r            = 1.0f;
        m.color.g            = 0.2f;
        m.color.b            = 0.2f;
        m.color.a            = 0.8f;

        geometry_msgs::Point origin;
        origin.x = origin.y = origin.z = 0.0;
        m.points.push_back(origin);

        for (int step = 0; step <= ARC_STEPS; ++step)
        {
            const float angle = right_angle + (left_angle - right_angle) * step / ARC_STEPS;
            geometry_msgs::Point p;
            p.x = RADIUS * std::cos(angle);
            p.y = RADIUS * std::sin(angle);
            p.z = 0.0;
            m.points.push_back(p);
        }

        m.points.push_back(origin);
        out.markers.push_back(std::move(m));
    }

    return out;
}

// =============================================================================
// Filter pipeline helpers
// =============================================================================

// Transforms a LaserScan to a PointCloud2 in base_link frame.
// Returns false if TF is unavailable or the transform fails.
static bool transformScanToCloud(
    const sensor_msgs::LaserScan& scan,
    tf::TransformListener& tf_listener,
    laser_geometry::LaserProjection& projector,
    sensor_msgs::PointCloud2& cloud_out)
{
    sensor_msgs::PointCloud2 lidar_cloud;
    projector.projectLaser(scan, lidar_cloud);

    tf::StampedTransform transform;
    try {
        tf_listener.lookupTransform("base_link", scan.header.frame_id, scan.header.stamp, transform);
    } catch (const tf::TransformException&) {
        try {
            tf_listener.lookupTransform("base_link", scan.header.frame_id, ros::Time(0), transform);
        } catch (const tf::TransformException& e) {
            ROS_WARN_THROTTLE(2.0, "[YDLIDAR] TF exception: %s", e.what());
            return false;
        }
    }

    pcl_ros::transformPointCloud("base_link", transform, lidar_cloud, cloud_out);
    return true;
}

// Applies Statistical Outlier Removal to remove noisy points.
static pcl::PointCloud<pcl::PointXYZ>::Ptr applySORFilter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(input);
    sor.setMeanK(kSORMeanK);
    sor.setStddevMulThresh(kSORStddevThresh);
    sor.filter(*output);
    return output;
}

// Returns the base polygon rotated by stage_angle.
static std::vector<std::pair<float, float>> rotatePolygon(
    const std::vector<std::pair<float, float>>& base_polygon,
    float stage_angle)
{
    const float cos_a = std::cos(stage_angle);
    const float sin_a = std::sin(stage_angle);
    std::vector<std::pair<float, float>> rotated;
    rotated.reserve(base_polygon.size());
    for (const auto& pt : base_polygon) {
        rotated.emplace_back(
            pt.first * cos_a - pt.second * sin_a,
            pt.first * sin_a + pt.second * cos_a
        );
    }
    return rotated;
}

// Returns the 4 corners of the door filter square in the robot frame.
// Returns an empty vector if no door pose is available, if the door is farther than
// g_door_filter_max_distance, or if the TF lookup fails.
static std::vector<std::pair<float, float>> computeDoorPolygon(
    tf::TransformListener& tf_listener,
    const ros::Time& stamp)
{
    if (!g_has_door_pose) return {};

    tf::StampedTransform transform;
    try {
        tf_listener.lookupTransform("base_link", "map", stamp, transform);
    } catch (const tf::TransformException&) {
        try {
            tf_listener.lookupTransform("base_link", "map", ros::Time(0), transform);
        } catch (const tf::TransformException& e) {
            ROS_WARN_THROTTLE(2.0, "[YDLIDAR] Door TF exception: %s", e.what());
            return {};
        }
    }

    const tf::Point door_center(g_door_pose.x, g_door_pose.y, 0.0);
    const tf::Point door_center_robot = transform * door_center;
    const float cx = static_cast<float>(door_center_robot.x());
    const float cy = static_cast<float>(door_center_robot.y());

    if (std::hypot(cx, cy) > g_door_filter_max_distance)
    {
        g_has_door_pose = false;
        return {};
    }

    // Derive orientation in robot frame by transforming a point along the door direction
    const tf::Point door_dir(g_door_pose.x + std::cos(g_door_pose.theta),
                             g_door_pose.y + std::sin(g_door_pose.theta), 0.0);
    const tf::Point door_dir_robot = transform * door_dir;
    const float theta = std::atan2(
        static_cast<float>(door_dir_robot.y() - door_center_robot.y()),
        static_cast<float>(door_dir_robot.x() - door_center_robot.x()));

    const float half    = g_door_size * 0.5f;
    const float along_x =  std::cos(theta), along_y =  std::sin(theta);  // unit vector along the door
    const float normal_x = -std::sin(theta), normal_y = std::cos(theta); // unit vector perpendicular to the door

    return {
        {cx + half * (along_x + normal_x), cy + half * (along_y + normal_y)},
        {cx + half * (along_x - normal_x), cy + half * (along_y - normal_y)},
        {cx + half * (-along_x - normal_x), cy + half * (-along_y - normal_y)},
        {cx + half * (-along_x + normal_x), cy + half * (-along_y + normal_y)},
    };
}

// Iterates the SOR-filtered cloud, removing robot-body and conic-zone points.
// Applies downsampling (every kDownsampleStride-th point).
// Returns the surviving points as (x, y) in meters.
static std::pair<std::vector<float>, std::vector<float>> collectObstaclePoints(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::vector<std::pair<float, float>>& robot_polygon,
    float stage_angle,
    const std::vector<FilterZone>& zones,
    const std::vector<std::pair<float, float>>& door_polygon)
{
    std::vector<float> out_x, out_y;
    out_x.reserve(cloud->size() / kDownsampleStride + 1);
    out_y.reserve(cloud->size() / kDownsampleStride + 1);

    for (std::size_t i = 0; i < cloud->size(); i += kDownsampleStride) {
        const pcl::PointXYZ& pt = cloud->points[i];
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y)) continue;
        if (!robot_polygon.empty() && isPointInPolygon(pt.x, pt.y, robot_polygon)) continue;
        if (isPointInConicFilterArea(pt.x, pt.y, stage_angle, zones)) continue;
        if (!door_polygon.empty() && isPointInPolygon(pt.x, pt.y, door_polygon)) continue;
        out_x.push_back(pt.x);
        out_y.push_back(pt.y);
    }
    return {out_x, out_y};
}

// Builds a PointCloud2 message from parallel x/y arrays (coordinates in meters).
static sensor_msgs::PointCloud2 buildFilteredCloud(
    const std::vector<float>& pts_x,
    const std::vector<float>& pts_y,
    const std_msgs::Header& header)
{
    sensor_msgs::PointCloud2 msg;
    msg.header    = header;
    msg.height    = 1;
    msg.width     = pts_x.size();
    msg.is_dense  = true;

    sensor_msgs::PointCloud2Modifier mod(msg);
    mod.setPointCloud2FieldsByString(1, "xyz");
    mod.resize(pts_x.size());

    sensor_msgs::PointCloud2Iterator<float> ix(msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iy(msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iz(msg, "z");
    for (std::size_t i = 0; i < pts_x.size(); ++i, ++ix, ++iy, ++iz) {
        *ix = pts_x[i];
        *iy = pts_y[i];
        *iz = 0.0f;
    }
    return msg;
}

// Orchestrates the full filter pipeline and publishes the results.
static void filterAndPublish(
    const sensor_msgs::LaserScan& scan,
    ros::Publisher& filtered_pub,
    ros::Publisher& polygon_pub,
    ros::Publisher& filter_zones_pub,
    ros::Publisher& door_polygon_pub,
    tf::TransformListener& tf_listener,
    laser_geometry::LaserProjection& projector)
{
    const bool need_filtered  = filtered_pub.getNumSubscribers() > 0;
    const bool need_polygon   = polygon_pub.getNumSubscribers() > 0;
    const bool need_door_viz  = door_polygon_pub.getNumSubscribers() > 0;
    if (!need_filtered && !need_polygon && !need_door_viz) return;

    std_msgs::Header viz_hdr;
    viz_hdr.frame_id = "base_link";
    viz_hdr.stamp    = scan.header.stamp;

    // Door polygon is independent of cloud processing — compute once for viz and filtering
    std::vector<std::pair<float, float>> door_polygon;
    if (need_filtered || need_door_viz)
        door_polygon = computeDoorPolygon(tf_listener, scan.header.stamp);

    if (need_door_viz)
        door_polygon_pub.publish(makeDoorPolygonMarkers(door_polygon, viz_hdr));

    if (!need_filtered && !need_polygon) return;

    sensor_msgs::PointCloud2 raw_cloud;
    if (!transformScanToCloud(scan, tf_listener, projector, raw_cloud))
        return;

    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(raw_cloud, *pcl_cloud);
    const auto sor_cloud = applySORFilter(pcl_cloud);

    FilterConfig cfg;
    {
        std::lock_guard<std::mutex> lk(g_config_mutex);
        cfg = g_filter_config;
    }

    const float stage_angle = g_stage_angle.load();

    auto scaled_polygon = cfg.base_polygon;
    for (auto& pt : scaled_polygon) {
        pt.second *= cfg.lateral_scale;
    }
    const auto robot_polygon = rotatePolygon(scaled_polygon, stage_angle);

    if (need_polygon && !robot_polygon.empty()) {
        polygon_pub.publish(makePolygonMarkers(robot_polygon, viz_hdr));
    }

    if (filter_zones_pub.getNumSubscribers() > 0 && !cfg.filter_zones.empty()) {
        filter_zones_pub.publish(makeFilterZoneMarkers(cfg.filter_zones, stage_angle, viz_hdr));
    }

    if (!need_filtered) return;

    const auto [pts_x, pts_y] = collectObstaclePoints(sor_cloud, robot_polygon, stage_angle, cfg.filter_zones, door_polygon);

    std_msgs::Header hdr;
    hdr.frame_id = "base_link";
    hdr.stamp    = scan.header.stamp;
    filtered_pub.publish(buildFilteredCloud(pts_x, pts_y, hdr));
}

// =============================================================================
// ROS callbacks
// =============================================================================

void vehicleStateCallback(const cube_msgs::VehicleState& msg)
{
    g_stage_angle.store(static_cast<float>(msg.stageAngle));
    g_drive_mode.store(msg.driveMode);
}

void filterZonesCallback(const std_msgs::Float32MultiArray& msg)
{
    std::vector<FilterZone> zones;
    const auto& data = msg.data;
    // Data is interleaved: [center1_rad, hw1_rad, center2_rad, hw2_rad, ...]
    for (std::size_t i = 0; i + 1 < data.size(); i += 2) {
        zones.push_back({data[i], data[i + 1]});
    }
    std::lock_guard<std::mutex> lk(g_config_mutex);
    g_filter_config.filter_zones = std::move(zones);
    ROS_INFO("[YDLIDAR] Received %zu filter zone(s)", g_filter_config.filter_zones.size());
}

void cartLateralScaleCallback(const std_msgs::Float32& msg)
{
    std::lock_guard<std::mutex> lk(g_config_mutex);
    g_filter_config.lateral_scale = msg.data;
    ROS_DEBUG("[YDLIDAR] Lateral scale updated: %.3f", msg.data);
}

void doorPoseCallback(const geometry_msgs::Pose2D& msg)
{
    g_door_pose     = msg;
    g_has_door_pose = true;
}

void cartPolygonCallback(const geometry_msgs::Polygon& msg)
{
    std::vector<std::pair<float, float>> polygon;
    polygon.reserve(msg.points.size());
    for (const auto& pt : msg.points) {
        // Clamp front points to x=0: we don't filter in front of the robot
        polygon.emplace_back(pt.x > 0.0f ? 0.0f : pt.x, pt.y);
    }
    std::lock_guard<std::mutex> lk(g_config_mutex);
    g_filter_config.base_polygon = std::move(polygon);
    ROS_INFO("[YDLIDAR] Received cart polygon with %zu points", g_filter_config.base_polygon.size());
}

bool stop_scan(std_srvs::Empty::Request& /*req*/, std_srvs::Empty::Response& /*res*/)
{
    is_paused = true;
    ROS_DEBUG("Stop scan");
    return laser.turnOff();
}

bool start_scan(std_srvs::Empty::Request& /*req*/, std_srvs::Empty::Response& /*res*/)
{
    is_paused   = false;
    retry_count = 0;
    ROS_DEBUG("Start scan");
    return laser.turnOn();
}

// Attempts to reinitialise the lidar after a failure.
// Returns false when the maximum number of attempts is reached.
bool restartLidarSession(ros::Time& last_restart_time, int& attempts)
{
    if (attempts >= kMaxRestartAttempts) return false;

    ++attempts;
    ROS_WARN("[YDLIDAR] Restarting lidar... attempt %d/%d", attempts, kMaxRestartAttempts);

    laser.turnOff();
    laser.disconnecting();
    ros::Duration(0.5).sleep();

    if (!laser.initialize()) {
        ROS_ERROR("[YDLIDAR] Re-init failed: %s", laser.DescribeError());
        last_restart_time = ros::Time::now();
        return true;
    }
    if (!laser.turnOn()) {
        ROS_ERROR("[YDLIDAR] turnOn failed: %s", laser.DescribeError());
        last_restart_time = ros::Time::now();
        return true;
    }

    ROS_INFO("[YDLIDAR] Restart successful!");
    last_restart_time = ros::Time::now();
    return true;
}

// =============================================================================
// Lidar hardware parameter setup
// =============================================================================

static void setupLidarHardwareParams(ros::NodeHandle& nh)
{
    // String params
    auto setStr = [&](const char* param, const char* def, int prop) {
        std::string val;
        nh.param<std::string>(param, val, def);
        laser.setlidaropt(prop, val.c_str(), val.size());
    };
    setStr("port",         "/dev/ydlidar", LidarPropSerialPort);
    setStr("ignore_array", "",             LidarPropIgnoreArray);

    // Int params
    auto setInt = [&](const char* param, int def, int prop) {
        int val = def;
        nh.param<int>(param, val, def);
        laser.setlidaropt(prop, &val, sizeof(int));
    };
    setInt("baudrate",            230400,              LidarPropSerialBaudrate);
    setInt("lidar_type",          TYPE_TRIANGLE,       LidarPropLidarType);
    setInt("device_type",         YDLIDAR_TYPE_SERIAL, LidarPropDeviceType);
    setInt("sample_rate",         9,                   LidarPropSampleRate);
    setInt("abnormal_check_count",4,                   LidarPropAbnormalCheckCount);
    setInt("intensity_bit",       10,                  LidarPropIntenstiyBit);

    // Bool params
    auto setBool = [&](const char* param, bool def, int prop) {
        bool val = def;
        nh.param<bool>(param, val, def);
        laser.setlidaropt(prop, &val, sizeof(bool));
    };
    setBool("resolution_fixed",   true,  LidarPropFixedResolution);
    setBool("reversion",          true,  LidarPropReversion);
    setBool("inverted",           true,  LidarPropInverted);
    setBool("auto_reconnect",     true,  LidarPropAutoReconnect);
    setBool("isSingleChannel",    false, LidarPropSingleChannel);
    setBool("intensity",          false, LidarPropIntenstiy);
    setBool("support_motor_dtr",  false, LidarPropSupportMotorDtrCtrl);

    // Float params
    auto setFloat = [&](const char* param, float def, int prop) {
        float val = def;
        nh.param<float>(param, val, def);
        laser.setlidaropt(prop, &val, sizeof(float));
    };
    setFloat("angle_max",  180.0f, LidarPropMaxAngle);
    setFloat("angle_min", -180.0f, LidarPropMinAngle);
    setFloat("range_max",   16.0f, LidarPropMaxRange);
    setFloat("range_min",    0.1f, LidarPropMinRange);
    setFloat("frequency",   10.0f, LidarPropScanFrequency);
}

// =============================================================================
// Raw scan building (SDK → ROS messages)
// =============================================================================

static sensor_msgs::LaserScan buildScanMsg(
    const LaserScan& sdk_scan,
    const std::string& frame_id,
    bool invalid_range_is_inf)
{
    sensor_msgs::LaserScan msg;
    msg.header.stamp.sec  = sdk_scan.stamp / 1000000000ul;
    msg.header.stamp.nsec = sdk_scan.stamp % 1000000000ul;
    msg.header.frame_id   = frame_id;

    msg.angle_min       = sdk_scan.config.min_angle;
    msg.angle_max       = sdk_scan.config.max_angle;
    msg.angle_increment = sdk_scan.config.angle_increment;
    msg.scan_time       = sdk_scan.config.scan_time;
    msg.time_increment  = sdk_scan.config.time_increment;
    msg.range_min       = sdk_scan.config.min_range;
    msg.range_max       = sdk_scan.config.max_range;

    const int size = static_cast<int>(
        (sdk_scan.config.max_angle - sdk_scan.config.min_angle) /
         sdk_scan.config.angle_increment + 1);
    msg.ranges.resize(size, invalid_range_is_inf ? std::numeric_limits<float>::infinity() : 0.0f);
    msg.intensities.resize(size);

    for (std::size_t i = 0; i < sdk_scan.points.size(); ++i) {
        const auto& pt = sdk_scan.points[i];
        const int idx  = static_cast<int>(
            std::ceil((pt.angle - sdk_scan.config.min_angle) / sdk_scan.config.angle_increment));
        if (idx >= 0 && idx < size && pt.range >= sdk_scan.config.min_range) {
            msg.ranges[idx]      = pt.range;
            msg.intensities[idx] = pt.intensity;
        }
    }
    return msg;
}

static sensor_msgs::PointCloud buildPointCloudMsg(
    const LaserScan& sdk_scan,
    const std_msgs::Header& header,
    bool point_cloud_preservative)
{
    sensor_msgs::PointCloud msg;
    msg.header = header;
    msg.channels.resize(2);
    msg.channels[0].name = "intensities";
    msg.channels[1].name = "stamps";

    for (std::size_t i = 0; i < sdk_scan.points.size(); ++i) {
        const auto& pt = sdk_scan.points[i];
        const bool in_range = pt.range >= sdk_scan.config.min_range &&
                              pt.range <= sdk_scan.config.max_range;
        if (!point_cloud_preservative && !in_range) continue;

        geometry_msgs::Point32 p;
        p.x = pt.range * std::cos(pt.angle);
        p.y = pt.range * std::sin(pt.angle);
        msg.points.push_back(p);
        msg.channels[0].values.push_back(pt.intensity);
        msg.channels[1].values.push_back(static_cast<float>(i) * sdk_scan.config.time_increment);
    }
    return msg;
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ydlidar_ros_driver");
    ROS_INFO("YDLIDAR ROS Driver Version: %s", YDLIDAR_ROS_VERSION);

    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    // ---- Publishers ----
    ros::Publisher scan_pub           = nh.advertise<sensor_msgs::LaserScan>("scan", 1);
    ros::Publisher pc_pub             = nh.advertise<sensor_msgs::PointCloud>("point_cloud", 1);
    ros::Publisher scan_filtered_pub  = nh.advertise<sensor_msgs::PointCloud2>("filtered_pointcloud", 1);
    ros::Publisher polygon_pub        = nh.advertise<visualization_msgs::MarkerArray>("obstacle_polygon_visualization", 1);
    ros::Publisher filter_zones_pub   = nh.advertise<visualization_msgs::MarkerArray>("filter_zones_marker", 1);
    ros::Publisher door_polygon_pub   = nh.advertise<visualization_msgs::MarkerArray>("door_polygon_visualization", 1);

    // ---- Lidar hardware setup ----
    setupLidarHardwareParams(nh_private);

    std::string frame_id = "laser_frame";
    nh_private.param<std::string>("frame_id", frame_id, frame_id);

    bool invalid_range_is_inf    = false;
    bool point_cloud_preservative = false;
    nh_private.param<bool>("invalid_range_is_inf",    invalid_range_is_inf,    invalid_range_is_inf);
    nh_private.param<bool>("point_cloud_preservative", point_cloud_preservative, point_cloud_preservative);

    // ---- Filter setup ----
    // Config is provided by vehicle_interface via latched topics on startup.
    nh_private.param<float>("door_size",                g_door_size,                 2.0f);
    nh_private.param<float>("door_filter_max_distance", g_door_filter_max_distance,  2.5f);

    ros::Subscriber vehicle_state_sub  = nh.subscribe("/cube/data/vehicle_state",              1, vehicleStateCallback);
    ros::Subscriber cart_polygon_sub   = nh.subscribe("/cube/unit_config/cart_polygon",        1, cartPolygonCallback);
    ros::Subscriber filter_zones_sub   = nh.subscribe("/cube/unit_config/filter_zones",        1, filterZonesCallback);
    ros::Subscriber lateral_scale_sub  = nh.subscribe("/cube/unit_config/cart_lateral_scale",  1, cartLateralScaleCallback);
    ros::Subscriber door_pose_sub      = nh.subscribe("/detected_door_center",                 1, doorPoseCallback);

    tf::TransformListener      tf_listener;
    laser_geometry::LaserProjection projector;

    // ---- Services ----
    ros::ServiceServer stop_srv  = nh.advertiseService("stop_scan",  stop_scan);
    ros::ServiceServer start_srv = nh.advertiseService("start_scan", start_scan);

    // ---- Laser init ----
    bool running = laser.initialize();
    if (running) {
        running = laser.turnOn();
    } else {
        ROS_ERROR("%s", laser.DescribeError());
    }

    ros::Rate r(30);
    lastRestart = ros::Time::now();

    while (running && ros::ok()) {
        LaserScan sdk_scan;

        if (laser.doProcessSimple(sdk_scan)) {
            const auto scan_msg = buildScanMsg(sdk_scan, frame_id, invalid_range_is_inf);
            const auto pc_msg   = buildPointCloudMsg(sdk_scan, scan_msg.header, point_cloud_preservative);

            // Publish raw data first
            scan_pub.publish(scan_msg);
            pc_pub.publish(pc_msg);

            filterAndPublish(scan_msg, scan_filtered_pub, polygon_pub, filter_zones_pub, door_polygon_pub, tf_listener, projector);

        } else {
            if (!is_paused && !restartLidarSession(lastRestart, retry_count)) {
                ROS_FATAL("[YDLIDAR] Max restart attempts reached. Shutting down.");
                break;
            }
        }

        r.sleep();
        ros::spinOnce();
    }

    laser.turnOff();
    ROS_INFO("[YDLIDAR] Stopping...");
    laser.disconnecting();
    return 0;
}
