import React from "react";
import {
  Image,
  ScrollView,
  View,
  StyleSheet,
  SafeAreaView,
} from "react-native";
import shap1 from "@/assets/images/SHAP/shap 1.png";
import shap2 from "@/assets/images/SHAP/shap 2.png";
import shap3 from "@/assets/images/SHAP/shap 3.png";
import shap4 from "@/assets/images/SHAP/shap 4.png";
import shap5 from "@/assets/images/SHAP/shap 5.png";
import shap6 from "@/assets/images/SHAP/shap 6.png";
import shap7 from "@/assets/images/SHAP/shap 7.png";

import { ThemedText } from "@/components/ThemedText";

export default function XAI() {
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.contentContainer}>
        <ThemedText style={styles.title}>Explainable AI</ThemedText>
        <ThemedText style={styles.subtitle}>
          SHapley Additive exPlanations (SHAP) Analysis
        </ThemedText>

        {/* First Image and Caption */}
        <View style={styles.imageContainer}>
          <Image source={shap1} style={styles.image} />
          <ThemedText style={styles.caption}>
            Plot 1: This plot explains how feature X impacts the model's
            prediction.
          </ThemedText>

          <ThemedText style={styles.description}>
            This Bar graph shows the average impact of different features on the
            model&apos;s prediction for each class. The coloured segments of
            each bar shows the contribution of that feature to a specific class.
            For example : tBodyGyroJerk-entropy()-X and fBodyGyro-entropy()-X
            have the biggest impact on activities : Sitting and Standing.
          </ThemedText>
        </View>

        {/* Second Image and Caption */}
        <View style={styles.imageContainer}>
          <Image source={shap2} style={styles.image} />
          <ThemedText style={styles.caption}>
            Plot 2: This shows the overall feature importance based on SHAP
            values.
          </ThemedText>
          <ThemedText style={styles.description}>
            This scatter plot shows the impact of tBodyGyroJerk-entropy()-X on
            the model&apos;s output. Higher values of this feature lead to a
            higher SHAP value, meaning it increases the likelihood of certain
            predictions. The color bar shows that fBodyGyro-meanFreq()-X
            modulates this effect, with positive values of this feature leading
            to even stronger impacts.
          </ThemedText>
        </View>

        {/* Third Image and Caption */}
        <View style={styles.imageContainer}>
          <Image source={shap3} style={styles.image} />
          <ThemedText style={styles.caption}>
            Plot 3: A breakdown of feature contributions to the final
            prediction.
          </ThemedText>
          <ThemedText style={styles.description}>
            This waterfall plot explains how individuals features push the model
            output higher or lower compared to the base value by taking a single
            prediction. Each bar represents the contribution of that feature to
            the final prediction. Some have negative impact while some have
            positive impact on the final prediction.
          </ThemedText>
        </View>

        {/* Remaining Images and Captions */}
        <View style={styles.imageContainer}>
          <Image source={shap4} style={styles.image} />
          <ThemedText style={styles.caption}>
            Plot 4: A further breakdown of feature contributions.
          </ThemedText>
          <ThemedText style={styles.description}>
            This SHAP decision plot shows how different features contribute to a
            particular prediction. The plot illustrates that features like
            tBodyGyroJerk-entropy()-X and tBodyAccJerk-entropy()-X, with high
            values (represented by red color), push the model output towards a
            higher value. On the other hand, an unknown combination of features
            represented in purple pushes the output towards a lower value.
          </ThemedText>
        </View>

        <View style={styles.imageContainer}>
          <Image source={shap5} style={styles.image} />
          <ThemedText style={styles.caption}>
            Plot 5: A comparative plot for different model predictions.
          </ThemedText>
          <ThemedText style={styles.description}>
            This scatter plot shows the impact of features on the model&apos;s
            output. Features like tBodyGyroJerk-entropy()-X and
            tBodyAccJerk-entropy()-X generally have a positive impact, with
            higher feature values (red) leading to higher SHAP values and thus
            increased model output.
          </ThemedText>
        </View>

        <View style={styles.imageContainer}>
          <Image source={shap6} style={styles.image} />
          <ThemedText style={styles.caption}>
            Plot 6: Global feature importance in the model.
          </ThemedText>
          <ThemedText style={styles.description}>
            This SHAP bar plot reveals the global importance of different
            features in the model, ranked by their mean absolute SHAP values.
            The "Sum of 552 other features" contributes the most to the model's
            output overall, while features like tBodyGyroJerk-entropy()-X and
            tBodyAccJerk-entropy()-X have a relatively smaller but still
            noticeable impact. This suggests that many features collectively
            influence the model's predictions.
          </ThemedText>
        </View>

        <View style={styles.imageContainer}>
          <Image source={shap7} style={styles.image} />
          <ThemedText style={styles.caption}>
            Plot 7: How feature Z affects different predictions.
          </ThemedText>
          <ThemedText style={styles.description}>
            This scatter plots the relationship between feature values and their
            impact on output. For some feature with higher values seem to have
            positive impact while for some feature , higher value seem to have
            negative impact.
          </ThemedText>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  description: {
    textAlign: "justify",
    marginTop: 20,
    fontSize: 14,
  },
  container: {
    flex: 1,
    paddingTop: 32,
  },
  contentContainer: {
    padding: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 8,
    textAlign: "center",
  },
  subtitle: {
    fontSize: 18,
    marginBottom: 16,
    textAlign: "center",
  },
  imageContainer: {
    alignItems: "center",
    marginBottom: 24, // Added consistent spacing between images and captions
  },
  image: {
    width: "90%",
    height: undefined,
    aspectRatio: 1, // This will maintain the image's aspect ratio
    resizeMode: "contain",
    marginBottom: 8, // Add some space between the image and caption
  },
  caption: {
    fontSize: 14,
    color: "#555",
    marginTop: 8,
    textAlign: "center",
  },
});
