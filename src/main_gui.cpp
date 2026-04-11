#include <QApplication>
#include "image_gui.hpp"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
    ImageGUI window;
    window.show();
    
    return app.exec();
}
