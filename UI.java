import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.net.URLEncoder;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;

public class UI extends JFrame {

    private JTextField ageField, dailyUsageField, postsField, likesField, commentsField, messagesField;
    private JComboBox<String> genderComboBox, platformComboBox;
    private JLabel resultLabel;

    public UI() {
        setTitle("Social Media Emotion Predictor");
        setSize(500, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new GridBagLayout());

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.insets = new Insets(10, 10, 10, 10);

        // Input fields with labels
        gbc.gridx = 0;
        gbc.gridy = 0;
        add(new JLabel("Age:"), gbc);
        gbc.gridx = 1;
        ageField = new JTextField(15);
        add(ageField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 1;
        add(new JLabel("Gender:"), gbc);
        gbc.gridx = 1;
        genderComboBox = new JComboBox<>(new String[] {"Select", "Male", "Female", "Non-binary"});
        add(genderComboBox, gbc);

        gbc.gridx = 0;
        gbc.gridy = 2;
        add(new JLabel("Platform:"), gbc);
        gbc.gridx = 1;
        platformComboBox = new JComboBox<>(new String[] {"Select","Facebook" ,"Instagram", "Twitter", "Snapchat"});
        add(platformComboBox, gbc);

        gbc.gridx = 0;
        gbc.gridy = 3;
        add(new JLabel("Daily Usage Time (minutes):"), gbc);
        gbc.gridx = 1;
        dailyUsageField = new JTextField(15);
        add(dailyUsageField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 4;
        add(new JLabel("Posts Per Day:"), gbc);
        gbc.gridx = 1;
        postsField = new JTextField(15);
        add(postsField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 5;
        add(new JLabel("Likes Received Per Day:"), gbc);
        gbc.gridx = 1;
        likesField = new JTextField(15);
        add(likesField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 6;
        add(new JLabel("Comments Received Per Day:"), gbc);
        gbc.gridx = 1;
        commentsField = new JTextField(15);
        add(commentsField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 7;
        add(new JLabel("Messages Sent Per Day:"), gbc);
        gbc.gridx = 1;
        messagesField = new JTextField(15);
        add(messagesField, gbc);

        // Buttons for predict and clear actions
        JButton predictButton = new JButton("Predict Emotion");
        JButton clearButton = new JButton("Clear Fields");

        gbc.gridx = 0;
        gbc.gridy = 8;
        gbc.gridwidth = 2;
        add(predictButton, gbc);

        gbc.gridy = 9;
        add(clearButton, gbc);

        // Result display
        gbc.gridy = 10;
        resultLabel = new JLabel("Prediction Result: ");
        add(resultLabel, gbc);

        // Button action listeners
        predictButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (validateFields()) {
                    String prediction = predictEmotion(
                            Integer.parseInt(ageField.getText()),
                            genderComboBox.getSelectedItem().toString(),
                            platformComboBox.getSelectedItem().toString(),
                            Integer.parseInt(dailyUsageField.getText()),
                            Integer.parseInt(postsField.getText()),
                            Integer.parseInt(likesField.getText()),
                            Integer.parseInt(commentsField.getText()),
                            Integer.parseInt(messagesField.getText())
                    );
                    resultLabel.setText("Prediction Result: " + prediction);
                }
            }
        });

        clearButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                clearFields();
            }
        });

        // Styling for visual appeal
        getContentPane().setBackground(new Color(245, 245, 250));
        predictButton.setBackground(new Color(100, 150, 255));
        predictButton.setForeground(Color.WHITE);
        clearButton.setBackground(new Color(240, 90, 90));
        clearButton.setForeground(Color.WHITE);

        setVisible(true);
    }

    // Method to send a request to the server with URL-encoded data
    private String predictEmotion(int age, String gender, String platform, int dailyUsage, int posts, int likes, int comments, int messages) {
        try {
            // Create the URI and URL objects
            URI uri = new URI("http", null, "localhost", 5000, "/predict", null, null);
            URL url = uri.toURL();
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();

            // Set up the connection
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
            conn.setDoOutput(true);

            // Create the URL-encoded data
            StringBuilder postData = new StringBuilder();
            postData.append("age=").append(URLEncoder.encode(String.valueOf(age), "UTF-8"));
            postData.append("&gender=").append(URLEncoder.encode(gender, "UTF-8"));
            postData.append("&platform=").append(URLEncoder.encode(platform, "UTF-8"));
            postData.append("&daily_usage=").append(URLEncoder.encode(String.valueOf(dailyUsage), "UTF-8"));
            postData.append("&posts=").append(URLEncoder.encode(String.valueOf(posts), "UTF-8"));
            postData.append("&likes=").append(URLEncoder.encode(String.valueOf(likes), "UTF-8"));
            postData.append("&comments=").append(URLEncoder.encode(String.valueOf(comments), "UTF-8"));
            postData.append("&messages=").append(URLEncoder.encode(String.valueOf(messages), "UTF-8"));
            String encodedData = postData.toString();

            // Send the URL-encoded data
            try (OutputStream os = conn.getOutputStream()) {
                os.write(encodedData.getBytes("UTF-8"));
            }

            // Check for response
            int responseCode = conn.getResponseCode();
            if (responseCode != HttpURLConnection.HTTP_OK) {
                return "Error: Server responded with code " + responseCode;
            }

            // Read and return the response
            StringBuilder response = new StringBuilder();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                String responseLine;
                while ((responseLine = br.readLine()) != null) {
                    response.append(responseLine.trim());
                }
            }
            
            // Return the server's response
            return response.toString();

        } catch (Exception e) {
            e.printStackTrace();
            return "Error in prediction request";
        }
    }

    // Validation to ensure all fields are filled
    private boolean validateFields() {
        if (ageField.getText().isEmpty() || dailyUsageField.getText().isEmpty() || postsField.getText().isEmpty() ||
                likesField.getText().isEmpty() || commentsField.getText().isEmpty() || messagesField.getText().isEmpty()) {
            JOptionPane.showMessageDialog(this, "Please fill in all fields.", "Input Error", JOptionPane.WARNING_MESSAGE);
            return false;
        }

        if (genderComboBox.getSelectedItem().equals("Select") || platformComboBox.getSelectedItem().equals("Select")) {
            JOptionPane.showMessageDialog(this, "Please select a valid option for Gender and Platform.", "Input Error", JOptionPane.WARNING_MESSAGE);
            return false;
        }
        return true;
    }

    // Clear all input fields
    private void clearFields() {
        ageField.setText("");
        dailyUsageField.setText("");
        postsField.setText("");
        likesField.setText("");
        commentsField.setText("");
        messagesField.setText("");
        genderComboBox.setSelectedIndex(0);
        platformComboBox.setSelectedIndex(0);
        resultLabel.setText("Prediction Result: ");
    }

    public static void main(String[] args) {
        new UI();
    }
}
