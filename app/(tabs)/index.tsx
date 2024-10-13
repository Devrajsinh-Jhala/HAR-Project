import { useState, useEffect } from "react";
import { Alert, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { Accelerometer, Gyroscope, Magnetometer } from "expo-sensors";
import axios from "axios";
import { ThemedText } from "@/components/ThemedText";

export default function App() {
  const activityMapping: any = {
    1: "Walking",
    2: "Walking Upstairs",
    3: "Walking Downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying Down",
  };
  const [activityName, setActivityName] = useState("");
  const [accelData, setAccelData] = useState({ x: 0, y: 0, z: 0 });
  const [gyroData, setGyroData] = useState({ x: 0, y: 0, z: 0 });
  const [magData, setMagData] = useState({ x: 0, y: 0, z: 0 });
  const [recording, setRecording] = useState(false);
  const [collectedData, setCollectedData] = useState([]);
  const [initialized, setInitialized] = useState(false); // Track sensor initialization
  const [prediction, setPrediction] = useState(null);
  const INTERVAL_MS = 20; // 50Hz -> 20ms
  const RECORD_DURATION_MS = 2560; // 2.56 seconds

  let accelSub, gyroSub, magSub;

  // Set sensor update intervals to 20ms for 50Hz sampling and initialize once
  useEffect(() => {
    Accelerometer.setUpdateInterval(INTERVAL_MS);
    Gyroscope.setUpdateInterval(INTERVAL_MS);
    Magnetometer.setUpdateInterval(INTERVAL_MS);

    // Initialize sensors without collecting data yet
    accelSub = Accelerometer.addListener(() => {});
    gyroSub = Gyroscope.addListener(() => {});
    magSub = Magnetometer.addListener(() => {});

    setInitialized(true); // Sensors are initialized and ready
    // Clean up on unmount
    return () => {
      accelSub && accelSub.remove();
      gyroSub && gyroSub.remove();
      magSub && magSub.remove();
    };
  }, []);

  // Function to start recording
  const startRecording = () => {
    if (!initialized) {
      console.log("Sensors are not initialized yet.");
      return; // Don't proceed until sensors are ready
    }

    setCollectedData([]); // Clear previous data
    setRecording(true);

    const startTime = Date.now();

    // Subscribe to accelerometer
    accelSub = Accelerometer.addListener((data) => {
      setAccelData(data);
      collectData(data, "accelerometer", startTime);
    });

    // Subscribe to gyroscope
    gyroSub = Gyroscope.addListener((data) => {
      setGyroData(data);
      collectData(data, "gyroscope", startTime);
    });

    // Subscribe to magnetometer
    magSub = Magnetometer.addListener((data) => {
      setMagData(data);
      collectData(data, "magnetometer", startTime);
    });

    // Stop recording after 2.56 seconds (2560 ms)
    setTimeout(stopRecording, RECORD_DURATION_MS);
  };

  // Collect sensor data
  const collectData = (data, sensorType, startTime) => {
    const currentTime = Date.now();
    if (currentTime - startTime >= RECORD_DURATION_MS) {
      return; // Stop collecting data once the duration is exceeded
    }

    const sensorData = {
      sensor: sensorType,
      timestamp: currentTime - startTime, // Relative time
      x: data.x,
      y: data.y,
      z: data.z,
    };

    setCollectedData((prev) => [...prev, sensorData]);
  };

  // Stop recording and unsubscribe from sensors
  const stopRecording = () => {
    setRecording(false);
    if (accelSub) accelSub.remove();
    if (gyroSub) gyroSub.remove();
    if (magSub) magSub.remove();

    fetchPrediction(); // Ready to send data to backend
  };

  const fetchPrediction = async () => {
    try {
      console.log("Sending data to backend...");
      console.log("Sensor data", collectedData);
      // console.log("Collected Data:", collectedData); // To ensure the data is formatted correctly before sending

      const response = await axios.post(
        "http://10.60.16.66:8000/predict/",
        { sensor_data: collectedData },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      console.log("Data Sent", response.status);
      console.log("Received Response:", response.data.prediction.prediction); // Log the response
      setPrediction(response.data.prediction.prediction); // Use the backend response for prediction
      setActivityName(activityMapping[response.data.prediction.prediction]);
    } catch (error: any) {
      console.log("Request Failed: ", error.message); // Log the error in case of failure
      Alert.alert("Error", "Prediction failed: " + error.message);
    }
  };

  return (
    <View style={styles.container}>
      <ThemedText style={styles.title}>Group-E Project</ThemedText>
      <ThemedText style={styles.title}>Human Activity Recognition</ThemedText>

      {/* Subtitle */}
      <ThemedText style={styles.subtitle}>
        Using your phone's accelerometer, gyroscope, and magnetometer data, we
        analyze and predict daily activities such as walking, laying, or
        sitting. This app is developed with React Native as frontend and Flask
        as backend.
      </ThemedText>
      <View style={styles.buttonContainer}>
        <TouchableOpacity
          onPress={recording ? null : startRecording}
          style={styles.button}
        >
          <Text>{recording ? "Recording..." : "Start Recording"}</Text>
        </TouchableOpacity>
      </View>
      <ThemedText style={styles.text}>
        Prediction: {activityName || "None"}
      </ThemedText>
    </View>
  );
}

const styles = StyleSheet.create({
  title: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 10,
    textAlign: "center",
  },
  subtitle: {
    fontSize: 16,
    marginBottom: 30,
    textAlign: "center",
  },
  container: {
    flex: 1,
    justifyContent: "center",
    paddingHorizontal: 20,
  },
  text: {
    textAlign: "center",
  },
  buttonContainer: {
    flexDirection: "row",
    alignItems: "stretch",
    marginTop: 15,
  },
  button: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#eee",
    padding: 10,
  },
});
