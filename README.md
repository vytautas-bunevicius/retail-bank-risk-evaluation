# Revolut-style Loan Application Form

## Overview

This project implements a modern, user-friendly loan application form inspired by Revolut's design aesthetic. It features a multi-step form with real-time validation, progress tracking, and a sleek UI. The form is designed to collect necessary information from loan applicants and submit it to a backend service for processing.

## Features

- Multi-step form with progress tracking
- Real-time form validation
- Field highlighting for required and invalid inputs
- Clickable progress bubbles for navigation
- Responsive design for various screen sizes
- AJAX submission to backend service
- Modal display for application results

## Technologies Used

- HTML5
- CSS3
- JavaScript (ES6+)
- jQuery (for AJAX requests)

## File Structure

```
loan-application/
│
├── index.html
├── styles.css
├── script.js
└── README.md
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/loan-application.git
   ```

2. Navigate to the project directory:
   ```
   cd loan-application
   ```

3. Open `index.html` in a web browser to view the form.

## Usage

The loan application form consists of three steps:

1. Personal Information
2. Financial Information
3. Loan Details

Users must complete all required fields in each step before proceeding to the next. The progress bar at the top of the form indicates the current step and overall progress.

### Form Validation

- Real-time validation is performed as users fill out the form
- Error messages appear below invalid fields
- Users cannot proceed to the next step or submit the form until all required fields are correctly filled

### Submission

Upon completing all steps and clicking the "Submit Application" button, the form data is sent to the backend service (`/predict` endpoint) for processing. The result is displayed in a modal dialog.

## Customization

### Styling

The form uses CSS variables for easy customization. To modify the color scheme, edit the following variables in the `<style>` section of `index.html`:

```css
:root {
    --background-color: #EEECE2;
    --card-color: #FFFFFF;
    --text-color: #000000;
    --primary-color: #CC7B5C;
    --secondary-color: #9C8AA5;
    --accent-color: #91A694;
    --border-color: #E5E4DF;
    --progress-color: #91A694;
    --error-color: #FF6B6B;
}
```

### Form Fields

To add or remove form fields:

1. Modify the HTML structure within the `<form>` element in `index.html`
2. Update the `validateStep` function in `script.js` to include any new validation rules
3. Adjust the `submitApplication` function to handle new form data if necessary

## Backend Integration

The form is designed to work with a backend service that accepts POST requests to the `/predict` endpoint. Ensure your backend service:

1. Accepts JSON payload
2. Returns a JSON response with at least the following fields:
   - `loan_approval`: "Approved" or "Rejected"
   - `monthly_payment`: (number) if approved

## Browser Compatibility

This form is compatible with modern web browsers, including:

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
