import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class EduProSegmentation:

    def __init__(self, users_path, courses_path, transactions_path):

        self.users_df = pd.read_csv(users_path)
        self.courses_df = pd.read_csv(courses_path)
        self.transactions_df = pd.read_csv(transactions_path)

        self.learner_profiles = None
        self.clusters = None

    def create_learner_profiles(self):

        merged_df = (
            self.transactions_df
            .merge(self.courses_df, on="CourseID", how="left")
            .merge(self.users_df, on="UserID", how="left")
        )

        learner_agg = (
            merged_df.groupby("UserID")
            .agg(
                TotalCoursesEnrolled=("CourseID", "nunique"),
                TotalSpending=("Amount", "sum"),
                AvgSpending=("Amount", "mean"),
                Age=("Age", "first"),
                Gender=("Gender", "first")
            )
            .reset_index()
        )

        engagement = self._calculate_engagement_features(merged_df)
        preference = self._calculate_preference_features(merged_df)
        behavioral = self._calculate_behavioral_features(merged_df)

        self.learner_profiles = (
            learner_agg
            .merge(engagement, on="UserID", how="left")
            .merge(preference, on="UserID", how="left")
            .merge(behavioral, on="UserID", how="left")
        )

        return self.learner_profiles

    def _calculate_engagement_features(self, df):

        engagement = (
            df.groupby("UserID")
            .agg(
                AvgCoursesPerCategory=(
                    "CourseCategory",
                    lambda x: (x.value_counts(normalize=True)).mean()
                ),
                EnrollmentFrequency=("TransactionDate", "count")
            )
            .reset_index()
        )

        return engagement

    def _calculate_preference_features(self, df):

        preference = (
            df.groupby("UserID")
            .agg(
                PreferredCategory=(
                    "CourseCategory",
                    lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"
                ),
                PreferredLevel=(
                    "CourseLevel",
                    lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"
                ),
                AvgCourseRating=("CourseRating", "mean")
            )
            .reset_index()
        )

        return preference

    def _calculate_behavioral_features(self, df):

        diversity = (
            df.groupby("UserID")["CourseCategory"]
            .nunique()
            .reset_index(name="DiversityScore")
        )

        course_level_dist = (
            df.groupby("UserID")["CourseLevel"]
            .value_counts()
            .unstack(fill_value=0)
        )

        beginner = (
            course_level_dist["Beginner"]
            if "Beginner" in course_level_dist.columns
            else pd.Series(0, index=course_level_dist.index)
        )

        advanced = (
            course_level_dist["Advanced"]
            if "Advanced" in course_level_dist.columns
            else pd.Series(0, index=course_level_dist.index)
        )

        behavioral = diversity.copy()

        behavioral["LearningDepthIndex"] = (
            beginner / (beginner + advanced + 1e-6)
        ).values

        return behavioral

    def preprocess_features(self):

        self.learner_profiles["Gender_encoded"] = (
            pd.factorize(self.learner_profiles["Gender"])[0]
        )

        numeric_features = [
            "TotalCoursesEnrolled",
            "TotalSpending",
            "AvgSpending",
            "Age",
            "AvgCoursesPerCategory",
            "EnrollmentFrequency",
            "AvgCourseRating",
            "DiversityScore",
            "LearningDepthIndex"
        ]

        self.learner_profiles[numeric_features] = (
            self.learner_profiles[numeric_features]
            .fillna(0)
        )

        scaler = StandardScaler()

        X = scaler.fit_transform(
            self.learner_profiles[numeric_features]
        )

        return X, numeric_features

    def determine_optimal_clusters(self, X, max_clusters=10):

        inertias = []
        silhouettes = []

        k_values = range(2, max_clusters + 1)

        for k in k_values:

            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            labels = model.fit_predict(X)

            inertias.append(model.inertia_)
            silhouettes.append(
                silhouette_score(X, labels)
            )

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(k_values, inertias, marker="o")
        plt.title("Elbow Method")
        plt.xlabel("k")
        plt.ylabel("Inertia")

        plt.subplot(1, 2, 2)
        plt.plot(k_values, silhouettes, marker="o")
        plt.title("Silhouette Scores")
        plt.xlabel("k")
        plt.ylabel("Score")

        plt.tight_layout()
        plt.savefig("optimal_clusters.png")
        plt.close()

        optimal_k = list(k_values)[np.argmax(silhouettes)]

        return optimal_k

    def segment_learners(self, X, n_clusters):

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters
        )

        kmeans_labels = kmeans.fit_predict(X)
        hierarchical_labels = hierarchical.fit_predict(X)

        self.learner_profiles["KMeans_Cluster"] = kmeans_labels
        self.learner_profiles["Hierarchical_Cluster"] = hierarchical_labels

        return kmeans_labels, hierarchical_labels

    def evaluate_clustering(self, X, labels):

        score = silhouette_score(X, labels)

        print(f"Silhouette Score: {score:.4f}")

        return score

    def generate_recommendations(self):

        recommendations = {}

        for cluster in sorted(
            self.learner_profiles["KMeans_Cluster"].unique()
        ):

            group = self.learner_profiles[
                self.learner_profiles["KMeans_Cluster"] == cluster
            ]

            preferred_category = (
                group["PreferredCategory"].mode().iloc[0]
            )

            avg_rating = (
                group["AvgCourseRating"].mean()
            )

            recommendations[f"Cluster_{cluster}"] = {
                "Learners": len(group),
                "PreferredCategory": preferred_category,
                "AverageRating": round(avg_rating, 2),
                "Recommendation":
                    f"Recommend high-rated {preferred_category} courses"
            }

        return recommendations


if __name__ == "__main__":

    segmentation = EduProSegmentation(
        "users.csv",
        "courses.csv",
        "transactions.csv"
    )

    profiles = segmentation.create_learner_profiles()

    print("\nLearner Profiles")
    print(profiles.head())

    X, features = segmentation.preprocess_features()

    optimal_k = segmentation.determine_optimal_clusters(X)

    print(f"\nOptimal Clusters: {optimal_k}")

    labels, _ = segmentation.segment_learners(
        X,
        optimal_k
    )

    segmentation.evaluate_clustering(X, labels)

    recommendations = segmentation.generate_recommendations()

    for cluster, rec in recommendations.items():
        print(f"\n{cluster}")
        print(rec)

    segmentation.learner_profiles.to_csv(
        "learner_segments.csv",
        index=False
    )

    print("\nSaved: learner_segments.csv")
