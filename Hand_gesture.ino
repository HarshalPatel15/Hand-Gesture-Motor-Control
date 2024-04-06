#include <ros.h>
#include <std_msgs/Float32.h>

ros::NodeHandle nh;

void percentageCallback(const std_msgs::Float32& msg) {
  float percentage = msg.data;
  if (percentage < 20)
  {
    analogWrite(9, 2.55 * 0);
    digitalWrite(3, HIGH);
  }
  if (percentage >= 20 && percentage <= 40)
  {
    analogWrite(9, 2.55 * 40);
    digitalWrite(3, HIGH);
  }
  if (percentage > 40 && percentage <= 60)
  {
    analogWrite(9, 2.55 * 60);
    digitalWrite(3, HIGH);
  }
  if (percentage > 60 && percentage <= 80)
  {
    analogWrite(9, 2.55 * 80);
    digitalWrite(3, HIGH);
  }
  if (percentage > 80 && percentage <= 100)
  {
    analogWrite(9, 2.55 * 100);
    digitalWrite(3, HIGH);
  }

}
  ros::Subscriber<std_msgs::Float32> sub("mega", &percentageCallback);

  void setup() {

    pinMode(9, OUTPUT);
    pinMode(3, OUTPUT);
    nh.initNode();
    nh.subscribe(sub);
  }

  void loop() {
    nh.spinOnce();
    delay(10);
  }
